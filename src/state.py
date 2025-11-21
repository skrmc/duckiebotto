"""Shared application state that links ROS nodes and the FastAPI UI."""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any, Dict, Optional, Tuple


DEFAULT_COLOR_REFS = {"yellow": "#f2c100", "white": "#ffffff"}


class SharedState:
    """Thread-safe state container."""

    _MANUAL_TTL = 0.6  # seconds

    def __init__(self, vehicle_name: str) -> None:
        if not vehicle_name:
            raise ValueError("VEHICLE_NAME must be provided")
        self.vehicle_name = vehicle_name
        self._lock = threading.Lock()
        self._mode = "stopped"
        self._traffic_light: str = "unknown"
        self._raw_jpeg: Optional[bytes] = None
        self._out_jpeg: Optional[bytes] = None
        self._last_image_time: Optional[dt.datetime] = None
        self._manual_command: Optional[Tuple[float, float]] = None
        self._manual_stamp: float = 0.0
        self._client_active: bool = False
        # --- New state flags ---
        self._manual_mode = False
        self._lane_follow_enabled = False
        self._obstacle_avoid_enabled = False
        self._traffic_light_enabled = False
        self._speed_limit = 1.0  # fraction of max auto speed
        self._lane_error: Optional[float] = None
        self._heading_error: Optional[float] = None
        self._lane_in_lane = False
        self._lane_pose_stamp = 0.0
        self._color_refs: Dict[str, str] = dict(DEFAULT_COLOR_REFS)
        self._color_tolerance_scale = 1.0
        # --- Obstacle avoidance state ---
        self._tof_distance: Optional[float] = None
        self._obstacle_detected: bool = False

    def _set_flag(self, attr: str, enabled: bool, disable_manual: bool = False) -> None:
        with self._lock:
            setattr(self, attr, enabled)
            if enabled and disable_manual:
                self._manual_mode = False

    # --------------------------- mutation helpers ---------------------------
    def set_manual_mode(self, enabled: bool) -> None:
        with self._lock:
            self._manual_mode = enabled
            if enabled:
                self._lane_follow_enabled = False
                self._obstacle_avoid_enabled = False
                self._traffic_light_enabled = False

    def set_lane_follow_enabled(self, enabled: bool) -> None:
        self._set_flag("_lane_follow_enabled", enabled, disable_manual=True)

    def set_obstacle_avoid_enabled(self, enabled: bool) -> None:
        self._set_flag("_obstacle_avoid_enabled", enabled)

    def set_traffic_light_enabled(self, enabled: bool) -> None:
        self._set_flag("_traffic_light_enabled", enabled, disable_manual=True)

    def set_all_auto_components(self, enabled: bool) -> None:
        with self._lock:
            self._lane_follow_enabled = enabled
            self._obstacle_avoid_enabled = enabled
            self._traffic_light_enabled = enabled
            if enabled:
                self._manual_mode = False

    def set_speed_limit(self, limit: float) -> None:
        """Limit is expressed as a fraction in [0.2, 1.0]."""
        limit = float(limit)
        limit = max(0.2, min(1.0, limit))
        with self._lock:
            self._speed_limit = limit

    def is_manual_mode(self) -> bool:
        with self._lock:
            return self._manual_mode

    def is_lane_follow_enabled(self) -> bool:
        with self._lock:
            return self._lane_follow_enabled

    def is_traffic_light_enabled(self) -> bool:
        with self._lock:
            return self._traffic_light_enabled

    def update_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode

    def set_traffic_light(self, value: str) -> None:
        with self._lock:
            self._traffic_light = value

    def update_images(
        self, raw_jpeg: Optional[bytes], out_jpeg: Optional[bytes]
    ) -> None:
        now = dt.datetime.utcnow()
        with self._lock:
            self._raw_jpeg = raw_jpeg
            self._out_jpeg = out_jpeg
            self._last_image_time = now

    def update_lane_pose(self, lane_pose: Optional[object]) -> None:
        with self._lock:
            if lane_pose is None:
                self._lane_error = None
                self._heading_error = None
                self._lane_in_lane = False
                self._lane_pose_stamp = 0.0
                return
            self._lane_error = float(getattr(lane_pose, "d", 0.0))
            self._heading_error = float(getattr(lane_pose, "phi", 0.0))
            self._lane_in_lane = bool(getattr(lane_pose, "in_lane", False))
            self._lane_pose_stamp = float(getattr(lane_pose, "stamp", time.monotonic()))

    def update_tof_distance(self, distance: Optional[float]) -> None:
        with self._lock:
            self._tof_distance = distance

    def set_obstacle_detected(self, detected: bool) -> None:
        with self._lock:
            self._obstacle_detected = detected

    def get_tof_distance(self) -> Optional[float]:
        with self._lock:
            return self._tof_distance

    def is_obstacle_detected(self) -> bool:
        with self._lock:
            return self._obstacle_detected

    def is_obstacle_avoid_enabled(self) -> bool:
        with self._lock:
            return self._obstacle_avoid_enabled

    def set_manual_command(self, left: float, right: float) -> None:
        with self._lock:
            self._manual_command = (float(left), float(right))
            self._manual_stamp = time.monotonic()

    def clear_manual_command(self) -> None:
        with self._lock:
            self._manual_command = None
            self._manual_stamp = 0.0

    def has_active_client(self) -> bool:
        with self._lock:
            return self._client_active

    def try_acquire_client(self) -> bool:
        with self._lock:
            if self._client_active:
                return False
            self._client_active = True
            return True

    def release_client(self) -> None:
        with self._lock:
            self._client_active = False
            self._manual_command = None
            self._manual_stamp = 0.0

    # --------------------------- read helpers -------------------------------
    def get_speed_limit(self) -> float:
        with self._lock:
            return self._speed_limit

    def get_raw_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._raw_jpeg

    def get_out_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._out_jpeg

    def get_manual_command(self) -> Optional[Tuple[float, float]]:
        now = time.monotonic()
        with self._lock:
            if self._manual_command is None:
                return None
            if now - self._manual_stamp > self._MANUAL_TTL:
                self._manual_command = None
                return None
            return self._manual_command

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            stamp = self._last_image_time.isoformat() if self._last_image_time else None
            manual = (
                None
                if self._manual_command is None
                else {"left": self._manual_command[0], "right": self._manual_command[1]}
            )
            is_auto_active = any(
                (
                    self._lane_follow_enabled,
                    self._traffic_light_enabled,
                    self._obstacle_avoid_enabled,
                )
            )
            return {
                "vehicle": self.vehicle_name,
                "enabled": is_auto_active or self._manual_mode,
                "manual_mode": self._manual_mode,
                "lane_follow_enabled": self._lane_follow_enabled,
                "obstacle_avoid_enabled": self._obstacle_avoid_enabled,
                "traffic_light_enabled": self._traffic_light_enabled,
                "mode": self._mode,
                "traffic_light": self._traffic_light,
                "lane_error": self._lane_error,
                "heading_error": self._heading_error,
                "lane_in_lane": self._lane_in_lane,
                "tof_distance": self._tof_distance,
                "obstacle_detected": self._obstacle_detected,
                "last_image_time": stamp,
                "manual_command": manual,
                "client_active": self._client_active,
                "speed_limit": self._speed_limit,
                "color_refs": dict(self._color_refs),
                "color_tolerance_scale": self._color_tolerance_scale,
            }

    def set_color_reference(self, color: str, value: str) -> None:
        normalized = self._normalize_hex(value)
        key = color.lower()
        if key not in self._color_refs:
            raise ValueError(f"Unsupported color reference '{color}'")
        if normalized is None:
            raise ValueError("Color value must be a HEX string like #FFA500")
        with self._lock:
            self._color_refs[key] = normalized

    def get_color_references(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._color_refs)

    def set_color_tolerance(self, scale: float) -> None:
        value = max(0.5, min(2.0, float(scale)))
        with self._lock:
            self._color_tolerance_scale = value

    def get_color_tolerance(self) -> float:
        with self._lock:
            return float(self._color_tolerance_scale)

    @staticmethod
    def _normalize_hex(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if not stripped.startswith("#"):
            stripped = f"#{stripped}"
        if len(stripped) != 7:
            return None
        try:
            int(stripped[1:], 16)
        except ValueError:
            return None
        return stripped.upper()
