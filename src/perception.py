"""Integrated perception pipeline for lane following and diagnostics."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from duckietown_msgs.msg import WheelEncoderStamped
from sensor_msgs.msg import CameraInfo

from dt_computer_vision.camera import CameraModel
from dt_computer_vision.camera.homography import HomographyToolkit
from dt_computer_vision.camera.types import ResolutionIndependentImagePoint
from dt_computer_vision.ground_projection import GroundProjector
from dt_computer_vision.line_detection import ColorRange, Detections, LineDetector
from dt_state_estimation.lane_filter import LaneFilterHistogram
from dt_state_estimation.lane_filter.types import Segment, SegmentColor, SegmentPoint


# Default tuning derived from dt-core reference configs
LINE_DETECTOR_CONFIG = {
    "img_size": (120, 160),
    "top_cutoff": 40,
    "colors": {
        "WHITE": {"low": [0, 0, 150], "high": [180, 100, 255]},
        "YELLOW": {"low": [25, 140, 100], "high": [45, 255, 255]},
    },
    "line_detector_parameters": {
        "canny_thresholds": [80, 200],
        "canny_aperture_size": 3,
        "dilation_kernel_size": 3,
        "hough_threshold": 2,
        "hough_min_line_length": 3,
        "hough_max_line_gap": 1,
    },
}

COLOR_CALIBRATION_TOLERANCES: Dict[str, Dict[str, int]] = {
    "YELLOW": {"h": 9, "s": 70, "v": 80},
    "WHITE": {"h": 20, "s": 30, "v": 30},
}

LANE_FILTER_CONFIG = {
    "mean_d_0": 0.0,
    "mean_phi_0": 0.0,
    "sigma_d_0": 0.1,
    "sigma_phi_0": 0.1,
    "delta_d": 0.02,
    "delta_phi": 0.1,
    "d_max": 0.3,  # 0.3
    "d_min": -0.3,  # -0.3
    "phi_min": -1.5,
    "phi_max": 1.5,
    "cov_v": 0.5,
    "linewidth_white": 0.05,
    "linewidth_yellow": 0.025,
    "lanewidth": 0.23,
    "min_max": 0.1,
    "sigma_d_mask": 0.1,
    "sigma_phi_mask": 0.2,
    "range_min": 0.2,
    "range_est": 0.45,
    "range_max": 0.6,
}

ENCODER_RESOLUTION = 135
WHEEL_BASELINE = 0.1
WHEEL_RADIUS = 0.0318


@dataclass
class LanePoseEstimate:
    d: float
    phi: float
    in_lane: bool
    stamp: float


@dataclass
class PerceptionResult:
    traffic_light: str
    processed_frame: np.ndarray
    lane_pose: Optional[LanePoseEstimate]
    white_line_visible: bool


@dataclass
class LanePipelineResult:
    processed_frame: np.ndarray
    lane_pose: Optional[LanePoseEstimate]
    white_visible: bool


class TrafficLightDetector:
    """Lightweight red traffic light classifier."""

    def detect(self, frame_bgr: np.ndarray) -> str:
        height, _ = frame_bgr.shape[:2]
        roi = frame_bgr[: max(1, int(height * 0.35)), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_hue = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        upper_hue = cv2.inRange(hsv, (160, 120, 120), (179, 255, 255))
        mask = cv2.bitwise_or(lower_hue, upper_hue)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity >= 0.65:
                return "red"
        return "clear"


class LaneFollowingPipeline:
    """Consolidated pipeline inspired by dt-core (detector -> projection -> filter)."""

    def __init__(
        self,
        vehicle_name: str,
        color_fetcher: Optional[Callable[[], Dict[str, str]]] = None,
        tolerance_fetcher: Optional[Callable[[], float]] = None,
    ) -> None:
        self.vehicle_name = vehicle_name
        params = LINE_DETECTOR_CONFIG["line_detector_parameters"]
        self._line_detector = LineDetector(**params)
        self._img_size = LINE_DETECTOR_CONFIG["img_size"]
        self._top_cutoff = int(LINE_DETECTOR_CONFIG["top_cutoff"])
        self._fallback_ranges = {
            color: ColorRange.fromDict(cfg)
            for color, cfg in LINE_DETECTOR_CONFIG["colors"].items()
        }
        self._color_order = ["YELLOW", "WHITE"]
        self._color_to_bgr = {
            "YELLOW": (0, 255, 255),
            "WHITE": (255, 255, 255),
        }
        self._color_fetcher = color_fetcher
        self._tolerance_fetcher = tolerance_fetcher

        self._camera_model: Optional[CameraModel] = None
        self._projector: Optional[GroundProjector] = None
        self._lane_filter = LaneFilterHistogram(
            encoder_resolution=ENCODER_RESOLUTION,
            wheel_baseline=WHEEL_BASELINE,
            wheel_radius=WHEEL_RADIUS,
            **LANE_FILTER_CONFIG,
        )
        self._left_ticks_ref: Optional[int] = None
        self._right_ticks_ref: Optional[int] = None
        self._pending_left = 0
        self._pending_right = 0
        self._last_lane_pose: Optional[LanePoseEstimate] = None
        self._arr_cutoff = np.array([0, self._top_cutoff, 0, self._top_cutoff])
        img_h, img_w = self._img_size
        self._arr_ratio = np.array(
            [
                1.0 / float(img_w),
                1.0 / float(img_h),
                1.0 / float(img_w),
                1.0 / float(img_h),
            ]
        )

    # ------------------------------------------------------------------
    def handle_camera_info(self, msg: CameraInfo) -> None:
        if self._camera_model is not None:
            return
        self._camera_model = CameraModel(
            width=msg.width,
            height=msg.height,
            K=np.reshape(msg.K, (3, 3)),
            D=np.reshape(msg.D, (len(msg.D),)),
            P=np.reshape(msg.P, (3, 4)),
        )
        homography = self._load_extrinsics()
        self._camera_model.H = homography
        self._projector = GroundProjector(self._camera_model)

    def handle_encoder(self, side: str, msg: WheelEncoderStamped) -> None:
        ticks = int(getattr(msg, "data", 0))
        if side == "left":
            if self._left_ticks_ref is None:
                self._left_ticks_ref = ticks
            delta = ticks - self._left_ticks_ref
            self._pending_left += delta
            self._left_ticks_ref = ticks
        else:
            if self._right_ticks_ref is None:
                self._right_ticks_ref = ticks
            delta = ticks - self._right_ticks_ref
            self._pending_right += delta
            self._right_ticks_ref = ticks

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> LanePipelineResult:
        if frame_bgr is None:
            raise ValueError("frame_bgr must not be None")

        resized = cv2.resize(
            frame_bgr,
            (self._img_size[1], self._img_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        working = resized[self._top_cutoff :, :, :]

        color_ranges = self._current_color_ranges()
        detections = self._line_detector.detect(working, color_ranges)
        lane_pose = self._update_lane_filter(detections)
        white_visible = bool(len(detections) >= 2 and detections[1].lines.size > 0)
        processed_frame = self._draw_overlay(frame_bgr, detections)
        return LanePipelineResult(
            processed_frame=processed_frame,
            lane_pose=lane_pose,
            white_visible=white_visible,
        )

    # ------------------------------------------------------------------
    def _update_lane_filter(
        self, detections: Iterable[Detections]
    ) -> Optional[LanePoseEstimate]:
        if self._projector is None:
            return self._last_lane_pose

        segments: List[Segment] = []
        for idx, det in enumerate(detections):
            if det.lines.size == 0:
                continue
            normalized = (det.lines + self._arr_cutoff) * self._arr_ratio
            color_name = self._color_order[idx]
            segments.extend(self._project_segments(normalized, color_name))

        if not segments:
            return self._last_lane_pose

        if self._pending_left or self._pending_right:
            self._lane_filter.predict(self._pending_left, self._pending_right)
            self._pending_left = 0
            self._pending_right = 0

        self._lane_filter.update(segments)
        estimate = self._extract_estimate()
        if estimate is None:
            return self._last_lane_pose

        d, phi, confidence = estimate
        in_lane = confidence > getattr(self._lane_filter, "min_max", 0.1)
        self._last_lane_pose = LanePoseEstimate(
            d=float(d), phi=float(phi), in_lane=bool(in_lane), stamp=time.time()
        )
        return self._last_lane_pose

    def _project_segments(
        self, normalized_lines: np.ndarray, color_name: str
    ) -> List[Segment]:
        if self._camera_model is None or self._projector is None:
            return []
        color = SegmentColor.WHITE
        if color_name == "YELLOW":
            color = SegmentColor.YELLOW

        projected: List[Segment] = []
        for x1, y1, x2, y2 in normalized_lines:
            p1 = self._project_point(x1, y1)
            p2 = self._project_point(x2, y2)
            if p1 is None or p2 is None:
                continue
            segment = Segment(points=[p1, p2], color=color)
            projected.append(segment)
        return projected

    def _project_point(self, x: float, y: float) -> Optional[SegmentPoint]:
        if self._camera_model is None or self._projector is None:
            return None
        rip = ResolutionIndependentImagePoint(x=float(x), y=float(y))
        pixel = self._camera_model.independent2pixel(rip)
        rect = self._camera_model.rectifier.rectify_pixel(pixel)
        vector = self._camera_model.pixel2vector(rect)
        ground = self._projector.vector2ground(vector)
        return SegmentPoint(x=float(ground.x), y=float(ground.y))

    def _extract_estimate(self) -> Optional[Tuple[float, float, float]]:
        if hasattr(self._lane_filter, "getEstimate"):
            try:
                d, phi = self._lane_filter.getEstimate()
            except ValueError:
                estimate = self._lane_filter.get_estimate()
                d, phi = estimate.get("d"), estimate.get("phi")
        else:
            estimate = self._lane_filter.get_estimate()
            if isinstance(estimate, dict):
                d = estimate.get("d")
                phi = estimate.get("phi")
            else:
                d, phi = estimate
        if d is None or phi is None:
            return None
        confidence = self._lane_filter.get_max()
        return float(d), float(phi), float(confidence)

    def _load_extrinsics(self) -> np.ndarray:
        folder = "/data/config/calibrations/camera_extrinsic"
        path = os.path.join(folder, f"{self.vehicle_name}.yaml")
        if not os.path.isfile(path):
            path = os.path.join(folder, "default.yaml")
        if not os.path.isfile(path):
            raise FileNotFoundError("No extrinsic calibration file found")
        matrix = HomographyToolkit.load_from_disk(path, return_date=False)
        return matrix.reshape((3, 3))

    def get_lane_pose(self) -> Optional[LanePoseEstimate]:
        return self._last_lane_pose

    def _draw_overlay(
        self, frame: np.ndarray, detections: Iterable[Detections]
    ) -> np.ndarray:
        overlay = frame.copy()
        if overlay.size == 0:
            return overlay
        height, width = overlay.shape[:2]
        scale_x = width / float(self._img_size[1])
        scale_y = height / float(self._img_size[0])
        thickness = max(1, min(width, height) // 200)
        offset = self._top_cutoff
        for idx, det in enumerate(detections):
            if det.lines.size == 0:
                continue
            color_name = self._color_order[idx]
            color = self._color_to_bgr.get(color_name, (0, 255, 255))
            for line in det.lines:
                x1, y1, x2, y2 = line
                y1 = (y1 + offset) * scale_y
                y2 = (y2 + offset) * scale_y
                x1 = x1 * scale_x
                x2 = x2 * scale_x
                pt1 = (int(np.clip(x1, 0, width - 1)), int(np.clip(y1, 0, height - 1)))
                pt2 = (int(np.clip(x2, 0, width - 1)), int(np.clip(y2, 0, height - 1)))
                cv2.line(overlay, pt1, pt2, color, thickness)
        return overlay

    def _current_color_ranges(self) -> List[ColorRange]:
        refs: Dict[str, str] = {}
        if self._color_fetcher is not None:
            try:
                refs = self._color_fetcher() or {}
            except Exception:
                refs = {}
        tol_scale = self._fetch_tolerance_scale()
        ranges: List[ColorRange] = []
        for color_name in self._color_order:
            key = color_name.lower()
            candidate = self._range_from_hex(color_name, refs.get(key), tol_scale)
            if candidate is None:
                candidate = self._fallback_ranges[color_name]
            ranges.append(candidate)
        return ranges

    def _range_from_hex(
        self, color_name: str, hex_value: Optional[str], tol_scale: float
    ) -> Optional[ColorRange]:
        hsv = self._hex_to_hsv(hex_value)
        if hsv is None:
            return None
        base_tol = COLOR_CALIBRATION_TOLERANCES.get(
            color_name, {"h": 10, "s": 60, "v": 60}
        )
        scale = max(0.2, min(4.0, float(tol_scale or 1.0)))
        tolerance = {
            "h": max(1, int(round(base_tol.get("h", 10) * scale))),
            "s": max(1, int(round(base_tol.get("s", 60) * scale))),
            "v": max(1, int(round(base_tol.get("v", 60) * scale))),
        }
        low = [
            int(max(0, hsv[0] - tolerance["h"])),
            int(max(0, hsv[1] - tolerance["s"])),
            int(max(0, hsv[2] - tolerance["v"])),
        ]
        high = [
            int(min(180, hsv[0] + tolerance["h"])),
            int(min(255, hsv[1] + tolerance["s"])),
            int(min(255, hsv[2] + tolerance["v"])),
        ]
        return ColorRange.fromDict({"low": low, "high": high})

    def _fetch_tolerance_scale(self) -> float:
        if self._tolerance_fetcher is None:
            return 1.0
        try:
            value = float(self._tolerance_fetcher())
        except Exception:
            return 1.0
        if not math.isfinite(value):
            return 1.0
        return max(0.2, min(4.0, value))

    @staticmethod
    def _hex_to_hsv(value: Optional[str]) -> Optional[Tuple[int, int, int]]:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        if not text.startswith("#"):
            text = f"#{text}"
        if len(text) != 7:
            return None
        try:
            r = int(text[1:3], 16)
            g = int(text[3:5], 16)
            b = int(text[5:7], 16)
        except ValueError:
            return None
        sample = np.uint8([[[b, g, r]]])
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)[0][0]
        return int(hsv[0]), int(hsv[1]), int(hsv[2])


class PerceptionProcessor:
    """High-level interface used by the ROS bridge."""

    def __init__(
        self,
        vehicle_name: str,
        color_fetcher: Optional[Callable[[], Dict[str, str]]] = None,
        tolerance_fetcher: Optional[Callable[[], float]] = None,
    ) -> None:
        self._traffic = TrafficLightDetector()
        self._lane_pipeline = LaneFollowingPipeline(
            vehicle_name, color_fetcher, tolerance_fetcher
        )

    def handle_camera_info(self, msg: CameraInfo) -> None:
        self._lane_pipeline.handle_camera_info(msg)

    def handle_encoder(self, side: str, msg: WheelEncoderStamped) -> None:
        self._lane_pipeline.handle_encoder(side, msg)

    def analyze(self, frame_bgr: np.ndarray) -> PerceptionResult:
        traffic_light = self._traffic.detect(frame_bgr)
        lane_result = self._lane_pipeline.process(frame_bgr)
        return PerceptionResult(
            traffic_light=traffic_light,
            processed_frame=lane_result.processed_frame,
            lane_pose=lane_result.lane_pose,
            white_line_visible=lane_result.white_visible,
        )

    def get_lane_pose(self) -> Optional[LanePoseEstimate]:
        return self._lane_pipeline.get_lane_pose()
