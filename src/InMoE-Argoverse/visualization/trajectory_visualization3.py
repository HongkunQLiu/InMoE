import os.path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

import sys
# sys.path.append("/mnt/ve_share2/zy/HIVT")
# from argoverse_api.argoverse.map_representation.map_api import ArgoverseMap
# from av2.map.map_api import ArgoverseStaticMap
#
from typing import Callable, Dict, List, Optional, Tuple, Union
from typing import Final, List, Optional, Sequence, Set, Tuple
# from av2.utils.typing import NDArrayFloat, NDArrayInt
from pathlib import Path

_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"
_DRIVABLE_AREA_COLOR: Final[str] = "white"       # 道路内
_BACKGROUND_COLOR: Final[str] = "lightgray"      # 道路外（非驾驶区域）

'''
过去轨迹 (Past Trajectory):

color="#ECA154"：这是过去轨迹的颜色，用浅橙色表示。
地面真实轨迹 (Ground Trut   ):

color="#d33e4c"：这是地面真实轨迹的颜色，用深红色表示。
预测轨迹 (Forecasted Trajectory):

color="#007672"：这是预测轨迹的颜色，用深绿色表示。
箭头和终点标记的颜色:

箭头和终点标记的颜色与相应轨迹的颜色一致，分别是橙色、红色和绿色。
'''


def plot_single_vehicle(
        sample_past_trajectory: np.ndarray,
        sample_groundtruth: np.ndarray,
        sample_forecasted_trajectories: List[np.ndarray],
        scenario_id,
        AgoMap, city_name
):
    plt.figure(facecolor=_BACKGROUND_COLOR)

    min_x = min(
        np.min(sample_past_trajectory[:, 0]),
        np.min(sample_groundtruth[:, 0]),
        np.min([np.min(traj[:, 0]) for traj in sample_forecasted_trajectories])
    )
    max_x = max(
        np.max(sample_past_trajectory[:, 0]),
        np.max(sample_groundtruth[:, 0]),
        np.max([np.max(traj[:, 0]) for traj in sample_forecasted_trajectories])
    )
    min_y = min(
        np.min(sample_past_trajectory[:, 1]),
        np.min(sample_groundtruth[:, 1]),
        np.min([np.min(traj[:, 1]) for traj in sample_forecasted_trajectories])
    )
    max_y = max(
        np.max(sample_past_trajectory[:, 1]),
        np.max(sample_groundtruth[:, 1]),
        np.max([np.max(traj[:, 1]) for traj in sample_forecasted_trajectories])
    )

    x_buffer = 10
    y_buffer = 10

    plt.xlim(min_x - x_buffer, max_x + x_buffer)
    plt.ylim(min_y - y_buffer, max_y + y_buffer)

    # 关键：xlim/ylim先设置，底色、地图再画！
    lane_ids = AgoMap.get_lane_ids_in_xy_bbox(
        (min_x + max_x) / 2, (min_y + max_y) / 2, city_name,
        max((max_x - min_x) / 2, (max_y - min_y) / 2) + 5
    )
    _plot_static_map_elements(AgoMap, lane_ids, city_name)

    # 下面才是轨迹等可视化
    plt.plot(
        sample_past_trajectory[:, 0],
        sample_past_trajectory[:, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=2,
        zorder=15,
        ls="--"
    )
    plt.plot(
        sample_groundtruth[:, 0],
        sample_groundtruth[:, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=2,
        zorder=20,
        ls="--"
    )

    for i, sample_forecasted_trajectory in enumerate(sample_forecasted_trajectories):
        plt.plot(
            sample_forecasted_trajectory[:, 0],
            sample_forecasted_trajectory[:, 1],
            color="#007672",
            label=f"Forecasted Trajectory {i + 1}",
            alpha=1,
            linewidth=2,
            zorder=20,
            ls="--"
        )
        plt.arrow(
            sample_forecasted_trajectory[-2, 0],
            sample_forecasted_trajectory[-2, 1],
            sample_forecasted_trajectory[-1, 0] - sample_forecasted_trajectory[-2, 0],
            sample_forecasted_trajectory[-1, 1] - sample_forecasted_trajectory[-2, 1],
            color="#007672",
            label="Forecasted Trajectory",
            alpha=1,
            linewidth=3,
            zorder=25,
            head_width=0.3,
            head_length=0.3
        )

    plt.arrow(
        sample_past_trajectory[-2, 0],
        sample_past_trajectory[-2, 1],
        sample_past_trajectory[-1, 0] - sample_past_trajectory[-2, 0],
        sample_past_trajectory[-1, 1] - sample_past_trajectory[-2, 1],
        color="#ECA154",
        label="Past Trajectory",
        alpha=1,
        linewidth=2.5,
        zorder=25,
        head_width=0.1,
    )
    plt.arrow(
        sample_groundtruth[-2, 0],
        sample_groundtruth[-2, 1],
        sample_groundtruth[-1, 0] - sample_groundtruth[-2, 0],
        sample_groundtruth[-1, 1] - sample_groundtruth[-2, 1],
        color="#d33e4c",
        label="Ground Truth",
        alpha=1,
        linewidth=3,
        zorder=30,
        head_width=0.1,
    )

    if not os.path.exists("xy4"):
        os.makedirs("xy4")
    plt.savefig(f'xy4/{scenario_id}.png')

def _plot_static_map_elements(
        AgoMap, lane_ids, city_name, show_ped_xings: bool = False
) -> None:
    ax = plt.gca()
    ax.set_facecolor(_BACKGROUND_COLOR)  # 灰色

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.fill(
        [xlim[0], xlim[1], xlim[1], xlim[0]],
        [ylim[0], ylim[0], ylim[1], ylim[1]],
        color=_BACKGROUND_COLOR,
        zorder=0
    )

    da_mat, city_to_image_se2 = AgoMap.get_rasterized_driveable_area(city_name)
    import cv2
    contours, _ = cv2.findContours(da_mat.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_to_city_se2 = np.linalg.inv(city_to_image_se2)

    for contour in contours:
        contour = contour.squeeze(1)
        if contour.shape[0] < 3:
            continue
        contour_hom = np.hstack([contour, np.ones((contour.shape[0], 1))])
        contour_city = (image_to_city_se2 @ contour_hom.T).T[:, :2]

        plt.fill(contour_city[:, 0], contour_city[:, 1], color="white", alpha=1.0, zorder=1)

    for lane_id in lane_ids:
        lane_boundary = AgoMap.get_lane_segment_polygon(lane_id, city_name)[:, :2]
        _plot_polylines(
            lane_boundary,
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )


def _plot_polylines(
        polylines: np.array,
        *,
        style: str = "-",
        line_width: float = 1.0,
        alpha: float = 1.0,
        color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    # for polyline in polylines:
    plt.plot(
            polylines[:, 0],
            polylines[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )


def _plot_polygons(
        polygons: np.array, *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)


