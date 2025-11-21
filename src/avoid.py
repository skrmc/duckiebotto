"""Offset-based obstacle avoidance planner.

This module adjusts the lane controller's `d_offset` to sidestep an obstacle,
then hands control back only after the right (white) line is visible again.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from perception import LanePoseEstimate


@dataclass
class AvoidanceStatus:
    """Lightweight status returned on every planner update."""

    d_offset: float
    mode: str
    active: bool


class AvoidancePlanner:
    """PID-offset based avoidance strategy with a safety return check."""

    def __init__(
        self,
        base_offset: float = 0.0,
        lane_width: float = 0.23,
        offset_rate: float = 0.35,
        white_timeout: float = 0.7,
        clearance_time: float = 0.5,
    ) -> None:
        self._base_offset = float(base_offset)
        self._avoid_offset = max(0.05, lane_width * 1.2)
        self._offset_rate = max(0.05, offset_rate)
        self._white_timeout = max(0.05, white_timeout)
        self._clearance_time = max(0.1, clearance_time)
        self._max_dt = 0.5
        self.reset()

    def reset(self) -> None:
        """Immediately stop any avoidance and return to the base offset."""
        now = time.monotonic()
        self._state = "idle"
        self._current_offset = self._base_offset
        self._target_offset = self._base_offset
        self._last_update = now
        self._last_white_seen = 0.0
        self._last_obstacle_seen = 0.0

    def update(
        self,
        enabled: bool,
        obstacle_detected: bool,
        lane_pose: Optional[LanePoseEstimate],
        white_line_visible: bool,
    ) -> AvoidanceStatus:
        """Advance the planner and return the current offset and mode."""
        now = time.monotonic()
        dt = min(self._max_dt, max(0.0, now - self._last_update))
        self._last_update = now

        if not enabled:
            self.reset()
            return AvoidanceStatus(
                d_offset=self._current_offset, mode="lane-follow", active=False
            )

        lane_ok = lane_pose is not None and lane_pose.in_lane

        if white_line_visible:
            self._last_white_seen = now
        if obstacle_detected:
            self._last_obstacle_seen = now

        obstacle_recent = obstacle_detected or (
            (now - self._last_obstacle_seen) < self._clearance_time
        )

        if (
            obstacle_recent
            and lane_ok
            and self._state
            in (
                "idle",
                "waiting",
                "returning",
            )
        ):
            self._state = "offsetting"
            self._target_offset = self._avoid_offset

        if self._state == "offsetting":
            self._target_offset = self._avoid_offset
            if self._at_target(self._avoid_offset):
                self._state = "holding"
        elif self._state == "holding":
            self._target_offset = self._avoid_offset
            if not obstacle_recent:
                self._state = "waiting"
        elif self._state == "waiting":
            self._target_offset = self._avoid_offset
            ready = self._white_ready(now) and lane_ok and not obstacle_recent
            if ready:
                self._state = "returning"
                self._target_offset = self._base_offset
        elif self._state == "returning":
            self._target_offset = self._base_offset
            if obstacle_recent and lane_ok:
                self._state = "offsetting"
                self._target_offset = self._avoid_offset
            elif self._at_target(self._base_offset):
                self._state = "idle"

        self._step_offset(dt)
        active = (
            self._state != "idle"
            or abs(self._current_offset - self._base_offset) > 1e-3
        )
        mode = "avoiding" if active else "lane-follow"
        return AvoidanceStatus(d_offset=self._current_offset, mode=mode, active=active)

    def _white_ready(self, now: float) -> bool:
        if self._last_white_seen <= 0.0:
            return False
        return (now - self._last_white_seen) <= self._white_timeout

    def _step_offset(self, dt: float) -> None:
        diff = self._target_offset - self._current_offset
        if abs(diff) < 1e-4:
            self._current_offset = self._target_offset
            return
        max_step = self._offset_rate * dt
        step = max(-max_step, min(max_step, diff))
        self._current_offset += step

    def _at_target(self, target: float) -> bool:
        return abs(self._current_offset - target) < 0.01
