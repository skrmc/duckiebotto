"""Closed-loop motion control utilities for the Duckiebot stack."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import rospy
from duckietown_msgs.msg import WheelEncoderStamped


@dataclass
class LaneControllerParams:
    """Hyper-parameters for the consolidated lane follower."""

    v_bar: float = 0.19
    k_d: float = -2.0
    k_theta: float = -3.0
    k_Id: float = -3.0
    k_Iphi: float = 0.0
    d_thres: float = 0.3  # 0.2615
    theta_thres_min: float = -0.5
    theta_thres_max: float = 0.75
    d_offset: float = 0.0
    omega_ff: float = 0.0
    d_resolution: float = 0.011
    phi_resolution: float = 0.051
    integral_bounds: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "d": {"top": 0.3, "bot": -0.3},
            "phi": {"top": 1.2, "bot": -1.2},
        }
    )
    stop_line_slowdown: Dict[str, float] = field(
        default_factory=lambda: {"start": 0.4, "end": 0.1}
    )


class LaneController:
    """PI controller that matches the dt-core lane controller behaviour."""

    def __init__(self, params: Optional[LaneControllerParams] = None) -> None:
        self.params = params or LaneControllerParams()
        self.reset()

    def update_parameters(self, params: LaneControllerParams) -> None:
        self.params = params

    def reset(self) -> None:
        self.d_I = 0.0
        self.phi_I = 0.0
        self.prev_d_err = 0.0
        self.prev_phi_err = 0.0

    def compute_control_action(
        self,
        d_err: float,
        phi_err: float,
        dt: Optional[float],
        wheels_cmd_exec: Optional[Tuple[float, float]],
        stop_line_distance: Optional[float],
    ) -> Tuple[float, float]:
        if dt is not None:
            self._integrate_errors(d_err, phi_err, dt)

        self.d_I = self._adjust_integral(
            d_err, self.d_I, self.params.integral_bounds["d"], self.params.d_resolution
        )
        self.phi_I = self._adjust_integral(
            phi_err,
            self.phi_I,
            self.params.integral_bounds["phi"],
            self.params.phi_resolution,
        )

        exec_left, exec_right = 0.0, 0.0
        if wheels_cmd_exec is not None:
            exec_left, exec_right = wheels_cmd_exec
        self._reset_if_needed(d_err, phi_err, exec_left, exec_right)

        omega = (
            self.params.k_d * d_err
            + self.params.k_theta * phi_err
            + self.params.k_Id * self.d_I
            + self.params.k_Iphi * self.phi_I
        )

        self.prev_d_err = d_err
        self.prev_phi_err = phi_err

        v = self._compute_velocity(stop_line_distance)
        return v, omega + self.params.omega_ff

    def _compute_velocity(self, stop_line_distance: Optional[float]) -> float:
        v_bar = self.params.v_bar
        if stop_line_distance is None:
            return v_bar

        d1 = self.params.stop_line_slowdown["start"]
        d2 = self.params.stop_line_slowdown["end"]
        if d1 <= d2:
            return v_bar
        c = (0.5 * (d1 - stop_line_distance) + (stop_line_distance - d2)) / (d1 - d2)
        v_new = v_bar * c
        return max(v_bar / 2.0, min(v_bar, v_new))

    def _integrate_errors(self, d_err: float, phi_err: float, dt: float) -> None:
        self.d_I += d_err * dt
        self.phi_I += phi_err * dt

    def _reset_if_needed(
        self, d_err: float, phi_err: float, exec_left: float, exec_right: float
    ) -> None:
        if d_err * self.prev_d_err < 0.0:
            self.d_I = 0.0
        if phi_err * self.prev_phi_err < 0.0:
            self.phi_I = 0.0
        if exec_left == 0.0 and exec_right == 0.0:
            self.d_I = 0.0
            self.phi_I = 0.0

    @staticmethod
    def _adjust_integral(
        error: float, integral: float, bounds: Dict[str, float], resolution: float
    ) -> float:
        top = bounds["top"]
        bot = bounds["bot"]
        if integral > top:
            return top
        if integral < bot:
            return bot
        return 0.0 if abs(error) < resolution else integral


@dataclass
class WheelState:
    velocity: float = 0.0
    distance: float = 0.0
    stamp: Optional[float] = None


class VelocityPID:
    """Simple PID regulator used for per-wheel velocity control."""

    def __init__(
        self, kp: float, ki: float, kd: float, integral_limit: float = 1.0
    ) -> None:
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def step(self, target: float, measurement: float, dt: float) -> float:
        error = target - measurement
        if dt <= 0.0:
            dt = 1e-3
        self._integral += error * dt
        self._integral = max(
            -self._integral_limit, min(self._integral_limit, self._integral)
        )
        derivative = 0.0
        if self._prev_error is not None:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error
        correction = (
            self._kp * error + self._ki * self._integral + self._kd * derivative
        )
        return target + correction


class MotionController:
    """Tracks desired linear/angular velocity using wheel encoder feedback."""

    def __init__(self, wheel_base: float = 0.10, max_wheel_speed: float = 1.0) -> None:
        self._wheel_base = max(0.02, wheel_base)
        self._max_wheel_speed = max_wheel_speed
        self._targets = {"linear": 0.0, "angular": 0.0}
        self._wheel_states = {
            "left": WheelState(),
            "right": WheelState(),
        }
        self._pid = {
            "left": VelocityPID(1.4, 0.1, 0.02, integral_limit=max_wheel_speed),
            "right": VelocityPID(1.4, 0.1, 0.02, integral_limit=max_wheel_speed),
        }
        self._last_control_stamp: Optional[float] = None
        self._lock = threading.Lock()

    # ------------------------- encoder integration -------------------------
    def update_encoder(self, side: str, msg: WheelEncoderStamped) -> None:
        stamp = self._extract_stamp(msg)
        velocity = self._extract_velocity(msg)
        position = self._extract_distance(msg)

        with self._lock:
            state = self._wheel_states[side]
            prev_distance = state.distance
            prev_stamp = state.stamp

            dt = None
            if stamp is not None and prev_stamp is not None:
                dt = max(0.0, stamp - prev_stamp)

            if (
                velocity is None
                and position is not None
                and prev_stamp is not None
                and dt
                and dt > 0.0
            ):
                velocity = (position - prev_distance) / dt

            if position is None and velocity is not None and dt and dt > 0.0:
                position = prev_distance + velocity * dt

            if velocity is not None:
                state.velocity = float(velocity)
            if position is not None:
                state.distance = float(position)
            state.stamp = stamp

    def _extract_stamp(self, msg: WheelEncoderStamped) -> Optional[float]:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        return stamp.to_sec() if stamp else rospy.Time.now().to_sec()

    @staticmethod
    def _extract_velocity(msg: WheelEncoderStamped) -> Optional[float]:
        for attr in ("velocity", "vel", "omega", "speed"):
            value = getattr(msg, attr, None)
            if value is not None:
                return float(value)
        return None

    @staticmethod
    def _extract_distance(msg: WheelEncoderStamped) -> Optional[float]:
        for attr in ("meter", "meters", "distance", "position", "pos"):
            value = getattr(msg, attr, None)
            if value is not None:
                return float(value)
        return None

    # ---------------------------- control loop -----------------------------
    def set_target(self, linear: float, angular: float) -> None:
        with self._lock:
            self._targets["linear"] = float(linear)
            self._targets["angular"] = float(angular)

    def stop(self) -> None:
        self.set_target(0.0, 0.0)
        for pid in self._pid.values():
            pid.reset()

    def step(self) -> Tuple[float, float]:
        with self._lock:
            now = rospy.Time.now().to_sec()
            dt = (
                max(0.001, now - self._last_control_stamp)
                if self._last_control_stamp
                else 0.05
            )
            self._last_control_stamp = now

            linear = self._targets["linear"]
            angular = self._targets["angular"]
            target_left = linear - (angular * self._wheel_base / 2.0)
            target_right = linear + (angular * self._wheel_base / 2.0)

            measured_left = self._wheel_states["left"].velocity
            measured_right = self._wheel_states["right"].velocity

            cmd_left = self._pid["left"].step(target_left, measured_left, dt)
            cmd_right = self._pid["right"].step(target_right, measured_right, dt)

            return (
                self._clamp(cmd_left, self._max_wheel_speed),
                self._clamp(cmd_right, self._max_wheel_speed),
            )

    def _clamp(self, value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    # --------------------------- odometry helpers -------------------------
    def get_wheel_distances(self) -> Tuple[float, float]:
        with self._lock:
            return (
                self._wheel_states["left"].distance,
                self._wheel_states["right"].distance,
            )

    def distance_delta(self, start: Tuple[float, float]) -> float:
        left, right = self.get_wheel_distances()
        return ((left - start[0]) + (right - start[1])) / 2.0

    def heading_delta(self, start: Tuple[float, float]) -> float:
        left, right = self.get_wheel_distances()
        current_delta = right - left
        start_delta = start[1] - start[0]
        return (current_delta - start_delta) / self._wheel_base
