"""ROS nodes for the Duckiebot application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
import threading
from duckietown_msgs.msg import WheelEncoderStamped, WheelsCmdStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Range

from avoid import AvoidancePlanner
from control import LaneController, LaneControllerParams
from perception import (
    LANE_FILTER_CONFIG,
    LanePoseEstimate,
    PerceptionProcessor,
    WHEEL_BASELINE,
)
from state import SharedState


@dataclass
class ControlCommand:
    mode: str
    left: float
    right: float


class DuckiebotNode:
    """Bridges the ROS world with the FastAPI UI."""

    def __init__(self, vehicle_name: str, state: SharedState) -> None:
        self.vehicle_name = vehicle_name
        self.state = state
        self._perception = PerceptionProcessor(
            vehicle_name, state.get_color_references, state.get_color_tolerance
        )
        self._traffic_light = "unknown"
        self._lane_params = LaneControllerParams()
        self._lane_controller = LaneController(self._lane_params)
        lane_width = float(LANE_FILTER_CONFIG.get("lanewidth", 0.23))
        self._avoidance = AvoidancePlanner(
            base_offset=self._lane_params.d_offset, lane_width=lane_width
        )
        self._last_lane_pose: Optional[LanePoseEstimate] = None
        self._last_lane_control_stamp: Optional[float] = None
        self._last_auto_cmd: Tuple[float, float] = (0.0, 0.0)
        self._baseline = WHEEL_BASELINE
        self._max_linear_speed = 0.5  # m/s used to normalize wheel commands
        self._white_line_visible = False
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._last_processed_seq = 0
        # --- Obstacle detection parameters ---
        self._tof_threshold = 0.5  # meters

        ns = f"/{vehicle_name}"
        camera_topic = f"{ns}/camera_node/image/compressed"
        camera_info_topic = f"{ns}/camera_node/camera_info"
        wheels_topic = f"{ns}/wheels_driver_node/wheels_cmd"
        left_encoder_topic = f"{ns}/left_wheel_encoder_node/tick"
        right_encoder_topic = f"{ns}/right_wheel_encoder_node/tick"
        tof_topic = f"{ns}/front_center_tof_driver_node/range"

        self._image_sub = rospy.Subscriber(
            camera_topic,
            CompressedImage,
            self._on_image,
            queue_size=1,
            buff_size=200000,
        )
        self._camera_info_sub = rospy.Subscriber(
            camera_info_topic, CameraInfo, self._on_camera_info, queue_size=1
        )
        self._wheels_pub = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        self._left_encoder_sub = rospy.Subscriber(
            left_encoder_topic, WheelEncoderStamped, self._on_left_encoder, queue_size=1
        )
        self._right_encoder_sub = rospy.Subscriber(
            right_encoder_topic,
            WheelEncoderStamped,
            self._on_right_encoder,
            queue_size=1,
        )
        self._tof_sub = rospy.Subscriber(
            tof_topic, Range, self._on_tof_range, queue_size=1
        )
        self._timer = rospy.Timer(rospy.Duration(0.05), self._on_control_timer)

        self._frame_worker_thread = threading.Thread(
            target=self._frame_worker, daemon=True
        )
        self._frame_worker_thread.start()

        rospy.loginfo("DuckiebotNode initialised for %s", vehicle_name)

    def _next_frame(self) -> Optional[Tuple[np.ndarray, int]]:
        with self._frame_lock:
            if (
                self._frame_seq == self._last_processed_seq
                or self._latest_frame is None
            ):
                self._frame_event.clear()
                return None
            return self._latest_frame.copy(), self._frame_seq

    def _store_processed_frames(
        self, raw_frame: np.ndarray, processed: np.ndarray
    ) -> None:
        raw_ok, raw_buf = cv2.imencode(".jpg", raw_frame)
        out_ok, out_buf = cv2.imencode(".jpg", processed)
        if raw_ok and out_ok:
            self.state.update_images(bytes(raw_buf), bytes(out_buf))
        self.state.set_traffic_light(self._traffic_light)
        self.state.update_lane_pose(self._last_lane_pose)

    def _clear_event_if_unchanged(self, seq: int) -> None:
        with self._frame_lock:
            if self._frame_seq == seq:
                self._frame_event.clear()

    # ------------------------------ callbacks ------------------------------
    def _on_image(self, msg: CompressedImage) -> None:
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if raw_frame is None:
            rospy.logwarn_throttle(5.0, "Failed to decode camera image")
            return

        with self._frame_lock:
            self._latest_frame = raw_frame.copy()
            self._frame_seq += 1
            self._frame_event.set()

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._perception.handle_camera_info(msg)
        if self._camera_info_sub is not None:
            self._camera_info_sub.unregister()
            self._camera_info_sub = None

    def _on_left_encoder(self, msg: WheelEncoderStamped) -> None:
        self._perception.handle_encoder("left", msg)

    def _on_right_encoder(self, msg: WheelEncoderStamped) -> None:
        self._perception.handle_encoder("right", msg)

    def _on_tof_range(self, msg: Range) -> None:
        distance = msg.range if msg.range >= 0.0 else None
        self.state.update_tof_distance(distance)
        if distance is not None and distance < self._tof_threshold:
            self.state.set_obstacle_detected(True)
        else:
            self.state.set_obstacle_detected(False)

    def _frame_worker(self) -> None:
        while not rospy.is_shutdown():
            self._frame_event.wait(timeout=1.0)
            if rospy.is_shutdown():
                break
            item = self._next_frame()
            if item is None:
                continue
            frame, seq = item
            result = self._perception.analyze(frame)
            self._traffic_light = result.traffic_light
            self._last_lane_pose = result.lane_pose
            self._white_line_visible = bool(result.white_line_visible)
            self._store_processed_frames(frame, result.processed_frame)
            self._last_processed_seq = seq
            self._clear_event_if_unchanged(seq)

    # ------------------------------- control --------------------------------
    def _on_control_timer(self, _event: rospy.timer.TimerEvent) -> None:
        cmd = self._compute_command()
        self.state.update_mode(cmd.mode)
        self._publish(cmd.left, cmd.right)

    def _compute_command(self) -> ControlCommand:
        # Manual mode has highest priority
        if self.state.is_manual_mode():
            self._avoidance.reset()
            self._lane_params.d_offset = 0.0
            left, right = self.state.get_manual_command() or (0.0, 0.0)
            return ControlCommand("manual", left, right)

        lane_follow_enabled = self.state.is_lane_follow_enabled()
        obstacle_avoid_enabled = self.state.is_obstacle_avoid_enabled()

        avoidance_status = None
        if lane_follow_enabled:
            avoidance_status = self._avoidance.update(
                enabled=obstacle_avoid_enabled,
                obstacle_detected=self.state.is_obstacle_detected(),
                lane_pose=self._last_lane_pose,
                white_line_visible=self._white_line_visible,
            )
            self._lane_params.d_offset = avoidance_status.d_offset
        else:
            self._avoidance.reset()
            self._lane_params.d_offset = 0.0

        traffic_stop = (
            self.state.is_traffic_light_enabled() and self._traffic_light == "red"
        )

        # Traffic light stops have highest priority (after manual)
        if traffic_stop:
            return ControlCommand("traffic-stop", 0.0, 0.0)

        # Lane following
        if lane_follow_enabled:
            mode = avoidance_status.mode if avoidance_status else "lane-follow"
            lane_cmd = self._lane_follow_command(mode)
            if lane_cmd is not None:
                return lane_cmd
            return ControlCommand("lane-lost", 0.0, 0.0)

        # Default is to stop if no other condition is met
        return ControlCommand("stopped", 0.0, 0.0)

    def _lane_follow_command(self, mode: str) -> Optional[ControlCommand]:
        pose = self._last_lane_pose
        if pose is None or not pose.in_lane:
            return None
        now = rospy.Time.now().to_sec()
        dt = None
        if self._last_lane_control_stamp is not None:
            dt = max(0.0, now - self._last_lane_control_stamp)
        self._last_lane_control_stamp = now

        params = self._lane_params
        d_err = pose.d - params.d_offset
        d_err = max(-params.d_thres, min(params.d_thres, d_err))
        phi_err = max(params.theta_thres_min, min(params.theta_thres_max, pose.phi))

        v, omega = self._lane_controller.compute_control_action(
            d_err, phi_err, dt, self._last_auto_cmd, None
        )
        v *= self.state.get_speed_limit()
        left, right = self._twist_to_wheels(v, omega)
        self._last_auto_cmd = (left, right)
        return ControlCommand(mode, left, right)

    def _twist_to_wheels(self, linear: float, angular: float) -> Tuple[float, float]:
        v_left = linear - 0.5 * self._baseline * angular
        v_right = linear + 0.5 * self._baseline * angular
        return (
            self._clamp(v_left / self._max_linear_speed, -1.0, 1.0),
            self._clamp(v_right / self._max_linear_speed, -1.0, 1.0),
        )

    def _publish(self, left: float, right: float) -> None:
        msg = WheelsCmdStamped()
        msg.vel_left = left
        msg.vel_right = right
        self._wheels_pub.publish(msg)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
