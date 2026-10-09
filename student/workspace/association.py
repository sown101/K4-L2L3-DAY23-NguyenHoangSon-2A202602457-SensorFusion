"""Measurement-to-track association via Mahalanobis gating and greedy matching.

Part F supplies the association stage shown in docs/HUONG_DAN_KY_THUAT.md §2.
Load ``kalman`` with ``load_workspace_module`` for innovation helpers and tracking parameters
for the chi-square gate.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np
from scipy.stats import chi2

from fusion_lab.workspace_loader import load_workspace_module
from fusion_lab.workspace_support import get_tracking_params

params = get_tracking_params()
kalman = load_workspace_module("kalman")


def mahalanobis_distance(track: Any, meas: Any) -> float:
    """Return squared Mahalanobis distance between a track and a measurement.

    Args:
        track: Track with ``x``, ``P``.
        meas: Measurement with ``sensor``.

    Returns:
        Scalar squared Mahalanobis distance.
    """
    H = meas.sensor.get_H(track.x)
    gamma = np.asarray(kalman.innovation(track.x, meas), dtype=float)
    S = np.asarray(kalman.innovation_covariance(track.P, meas, H), dtype=float)
    # vi: Giải S @ weighted_residual = gamma thay vì tính nghịch đảo S.
    weighted_residual = np.linalg.solve(S, gamma)
    return float((gamma.T @ weighted_residual).item())


def chi2_gate(mhd_sq: float, sensor: Any) -> bool:
    """Return True if squared Mahalanobis distance lies inside the chi-square gate.

    Args:
        mhd_sq: Squared Mahalanobis distance.
        sensor: Sensor with ``dim_meas``.

    Returns:
        True if inside gate.
    """
    threshold = chi2.ppf(params.gating_threshold, sensor.dim_meas)
    return bool(np.isfinite(mhd_sq) and 0 <= mhd_sq <= threshold)


def association_cost_matrix(
    track_list: Sequence[Any], meas_list: Sequence[Any]
) -> np.matrix:
    """Build gated costs, checking each sensor's visibility before projection.

    Args:
        track_list: Active tracks.
        meas_list: Measurements for this sensor pass.

    Returns:
        Cost matrix; ``np.inf`` for invisible tracks or rejected chi-square gates.
        Invisible pairs must never call the Mahalanobis/projection helpers.
    """
    costs = np.full((len(track_list), len(meas_list)), np.inf)
    for row, track in enumerate(track_list):
        for column, meas in enumerate(meas_list):
            # vi: Không chiếu camera hay tính MHD cho cặp ngoài FOV.
            if not meas.sensor.in_fov(track.x):
                continue
            distance = mahalanobis_distance(track, meas)
            if chi2_gate(distance, meas.sensor):
                costs[row, column] = distance
    return np.asmatrix(costs)


def pick_next_pair(
    association_matrix: np.matrix,
    unassigned_tracks: Sequence[Any],
    unassigned_meas: Sequence[Any],
) -> tuple[Any, Any, np.matrix, list[Any], list[Any]]:
    """Pick the minimum-cost track/measurement pair and shrink the association problem.

    Args:
        association_matrix: Current cost matrix.
        unassigned_tracks: Track objects still free.
        unassigned_meas: Measurement objects still free.

    Returns:
        Tuple (track, meas, new_matrix, remaining_tracks, remaining_meas).
        If no finite pair exists, return np.nan for track and meas and retain both lists.
    """
    costs = np.asarray(association_matrix, dtype=float)
    tracks = list(unassigned_tracks)
    measurements = list(unassigned_meas)
    finite = np.isfinite(costs)
    if not finite.any():
        return np.nan, np.nan, np.asmatrix(costs), tracks, measurements
    row, column = np.unravel_index(
        np.argmin(np.where(finite, costs, np.inf)), costs.shape
    )
    track = tracks.pop(row)
    meas = measurements.pop(column)
    remaining_costs = np.delete(np.delete(costs, row, axis=0), column, axis=1)
    return track, meas, np.asmatrix(remaining_costs), tracks, measurements


def associate_and_update(
    manager: Any,
    meas_list: Sequence[Any],
    filter_obj: Any,
    sensor: Any,
) -> None:
    """Greedy association loop with EKF updates and track management.

    Args:
        manager: Track manager (``track_list``, ``manage_tracks``, ...).
        meas_list: Lidar or camera measurements for this frame pass.
        filter_obj: Filter with ``predict`` / ``update``.
        sensor: Explicit lidar/camera pass sensor, including empty measurement frames.

    Returns:
        None; updates tracks in place and always finishes the lifecycle pass.
        Visibility is handled in the cost matrix, before pair removal. Camera
        updates refine state only; lidar hits alone increase existence scores.
    """
    unassigned_tracks = list(manager.track_list)
    unassigned_meas = list(meas_list)
    costs = association_cost_matrix(unassigned_tracks, unassigned_meas)
    while np.isfinite(costs).any():
        track, meas, costs, unassigned_tracks, unassigned_meas = pick_next_pair(
            costs, unassigned_tracks, unassigned_meas
        )
        filter_obj.update(track, meas)
        # vi: Platform chỉ ghi nhận hit cho LiDAR; camera không đổi score.
        manager.handle_updated_track(track, sensor)
    # vi: Lượt rỗng vẫn phải quản lý các track chưa được ghép theo sensor.
    manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)
