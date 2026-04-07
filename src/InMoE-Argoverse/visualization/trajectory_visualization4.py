import os.path

import matplotlib.pyplot as plt
import numpy as np

# sys.path.append("/mnt/ve_share2/zy/HIVT")
# from argoverse_api.argoverse.map_representation.map_api import ArgoverseMap
# from av2.map.map_api import ArgoverseStaticMap
#
from typing import Final, List
# from av2.utils.typing import NDArrayFloat, NDArrayInt
from matplotlib.patches import Polygon, RegularPolygon, Circle

from .vis_config_bright import canvas_config, road_line_config, road_edge_config, speed_bump_config, \
    crosswalk_config, lane_config, stop_sign_config, object_config, signal_config, driveway_config


_LANE_SEGMENT_COLOR: Final[str] = "#E0E0E0"
_DRIVABLE_AREA_COLOR: Final[str] = "white"       # 道路内
_BACKGROUND_COLOR: Final[str] = (255/255, 250/255, 224/255, 0.2)

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


def plot_obj_pose(
    obj_type: str,
    state: np.ndarray,
    ax: plt.Axes = None,
    facecolor: float = None,
    alpha: float = None,
):
    # state have feature [center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, valid]
    if state[-1] == 0:
        # Ignore invalid object
        return
    config = object_config[obj_type]
    if ax is None:
        ax = plt.gca()

    facecolor = config['facecolor'] if facecolor is None else facecolor
    alpha = config['alpha'] if alpha is None else alpha

    length = 4.0
    width = 2.0
    heading = state[6]

    vertices = np.array([
        [length / 3, width / 2, 1],
        [length / 2, 0, 1],  # add heading arrow
        [length / 3, -width / 2, 1],
        [-length / 2, -width / 2, 1],
        [-length / 2, width / 2, 1],
    ])

    pose = np.array([
        [np.cos(heading), -np.sin(heading), state[0]],
        [np.sin(heading), np.cos(heading), state[1]],
        [0, 0, 1],
    ])

    vertices_global = np.dot(pose, vertices.T).T[:, :2]
    p = Polygon(vertices_global, facecolor=facecolor, alpha=alpha, zorder=250)
    ax.add_patch(p)


def plot_signal(
        dynamic_map_infos: dict,
        t: int,
        ax: plt.Axes = None,
        linewidth: float = None,
        radius: float = None,
):
    if ax is None:
        ax = plt.gca()

    radius = signal_config['radius'] if radius is None else radius
    linewidth = signal_config['linewidth'] if linewidth is None else linewidth

    states_list = dynamic_map_infos['state'][t]
    stop_points_list = dynamic_map_infos['stop_point'][t]

    for states, stop_points in zip(states_list, stop_points_list):
        for signal_state, stop_point in zip(states, stop_points):
            if stop_point[-1] == 0:
                continue

            # override default config
            config = signal_config[signal_state]

            facecolor = config['facecolor']
            edgecolor = config['edgecolor']
            alpha = config['alpha']

            if config['shape'] == 'circle':
                p = Circle(
                    stop_point[:2],
                    radius=radius,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            elif config['shape'] == 'rectangle':
                p = RegularPolygon(
                    stop_point[:2],
                    radius=radius,
                    numVertices=4,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            elif config['shape'] == 'triangle':
                p = RegularPolygon(
                    stop_point[:2],
                    radius=radius,
                    numVertices=3,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            else:
                warnings.warn(f'Unknown shape {config["shape"]}')
                continue
            ax.add_patch(p)


def plot_single_vehicle(
        sample_past_trajectory: np.ndarray,  # 过去轨迹的坐标数组，形状为 (1, 50, 2)
        sample_groundtruth: np.ndarray,  # 地面真实轨迹的坐标数组，形状为 (1, 60, 2)
        sample_forecasted_trajectories: List[np.ndarray],  # 预测轨迹的列表，每个数组形状为 (6,60, 2)
        scenario_id,
        AgoMap,city_name
):
    plt.figure(facecolor=_BACKGROUND_COLOR)

    min_x = min(
        np.min(sample_past_trajectory[:, 0]),
        np.min(sample_groundtruth[:, 0]),
        np.min([np.min(traj[:, 0]) for traj in sample_forecasted_trajectories])
    )
    max_x = max(
        np.max(sample_past_trajectory[ :, 0]),
        np.max(sample_groundtruth[:, 0]),
        np.max([np.max(traj[:, 0]) for traj in sample_forecasted_trajectories])
    )
    min_y = min(
        np.min(sample_past_trajectory[ :, 1]),
        np.min(sample_groundtruth[ :, 1]),
        np.min([np.min(traj[:, 1]) for traj in sample_forecasted_trajectories])
    )
    max_y = max(
        np.max(sample_past_trajectory[ :, 1]),
        np.max(sample_groundtruth[ :, 1]),
        np.max([np.max(traj[:, 1]) for traj in sample_forecasted_trajectories])
    )

    x_buffer = 15
    y_buffer = 15

    vehicle_type = "TYPE_CYCLIST"
    if vehicle_type == "TYPE_CYCLIST":
        origin_color = 'yellow'
    elif vehicle_type == "TYPE_PEDESTRIAN":
        origin_color = 'pink'
    else:
        origin_color = 'red'

    plt.plot(sample_past_trajectory[0, 0], sample_past_trajectory[0, 1], markersize=4, marker='^', color='blue', zorder=100, alpha=0.5)
    plt.plot(sample_past_trajectory[-1, 0], sample_past_trajectory[-1, 1], markersize=8, marker='o', color=origin_color, zorder=251,
            alpha=0.5)
    plt.plot(sample_groundtruth[-1, 0], sample_groundtruth[-1, 1], markersize=6, marker='*', color='darkmagenta',
            zorder=100, alpha=0.5)

    plt.plot(sample_past_trajectory[:, 0], sample_past_trajectory[:, 1], color='xkcd:blue', linewidth=3, linestyle='-', alpha=0.6,
            zorder=100)

    plt.plot(sample_groundtruth[:, 0], sample_groundtruth[:, 1], color='xkcd:purple', linewidth=1.5, linestyle='-',
            alpha=0.4,
            zorder=10)


    # plt.plot(
    #     sample_past_trajectory[:, 0],
    #     sample_past_trajectory[:, 1],
    #     color="#ECA154",
    #     label="Past Trajectory",
    #     alpha=1,
    #     linewidth=2,
    #     zorder=15,
    #     ls="--"
    # )

    pt1 = sample_past_trajectory[-2]
    pt2 = sample_past_trajectory[-1]
    center_x, center_y = pt2[0], pt2[1]
    center_z = 0
    length, width, height = 4.0, 2.0, 1.5
    heading = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    velocity_x, velocity_y = pt2[0] - pt1[0], pt2[1] - pt1[1]
    state = np.array([center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, 1])
    plot_obj_pose('TYPE_VEHICLE', state, ax=plt.gca())

    # plt.plot(
    #     sample_groundtruth[:, 0],
    #     sample_groundtruth[ :, 1],
    #     color="#d33e4c",
    #     label="Ground Truth",
    #     alpha=1,
    #     linewidth=2,
    #     zorder=20,
    #     ls="--"
    # )

    pred_last = sample_forecasted_trajectories[:, -1, :]  # shape: (6, 2)
    gt_last = sample_groundtruth[-1, :]  # shape: (2,)

    # 计算每条轨迹的FDE（欧氏距离）
    fdes = np.linalg.norm(pred_last - gt_last, axis=1)  # shape: (6,)

    # 找到最小FDE的index
    min_fde_index = np.argmin(fdes)

    for i, sample_forecasted_trajectory in enumerate(sample_forecasted_trajectories):
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_forecasted_trajectory)))
        if i == min_fde_index:
            for dc in range(len(sample_forecasted_trajectory) - 1):
                plt.plot(sample_forecasted_trajectory[dc:dc + 2, 0], sample_forecasted_trajectory[dc:dc + 2, 1], linewidth=3,
                        color=colors[dc], zorder=9)
        else:
            for dc in range(len(sample_forecasted_trajectory) - 1):
                plt.plot(sample_forecasted_trajectory[dc:dc + 2, 0], sample_forecasted_trajectory[dc:dc + 2, 1], linewidth=3,
                        color=colors[dc], zorder=4)

        # plt.plot(
        #     sample_forecasted_trajectory[:, 0],
        #     sample_forecasted_trajectory[:, 1],
        #     color="#007672",
        #     label=f"Forecasted Trajectory {i + 1}",
        #     alpha=1,
        #     linewidth=2,
        #     zorder=20,
        #     ls="--"
        # )

        # Plot the end marker for forecasted trajectories
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

    # Plot the end marker for history
    # plt.arrow(
    #     sample_past_trajectory[ -2, 0],
    #     sample_past_trajectory[ -2, 1],
    #     sample_past_trajectory[ -1, 0] - sample_past_trajectory[-2, 0],
    #     sample_past_trajectory[ -1, 1] - sample_past_trajectory[ -2, 1],
    #     color="#ECA154",
    #     label="Past Trajectory",
    #     alpha=1,
    #     linewidth=2.5,
    #     zorder=25,
    #     head_width=0.1,
    # )

    # Plot the end marker for ground truth
    # plt.arrow(
    #     sample_groundtruth[ -2, 0],
    #     sample_groundtruth[ -2, 1],
    #     sample_groundtruth[ -1, 0] - sample_groundtruth[-2, 0],
    #     sample_groundtruth[ -1, 1] - sample_groundtruth[-2, 1],
    #     color="#d33e4c",
    #     label="Ground Truth",
    #     alpha=1,
    #     linewidth=3,
    #     zorder=30,
    #     head_width=0.1,
    # )
    # static_map_path = f"/mnt/ve_share2/zy/Argoverse_2_Motion_Forecasting_Dataset/raw/val/{scenario_id}/log_map_archive_{scenario_id}.json"
    # static_map_path = Path(static_map_path)
    # static_map = ArgoverseStaticMap.from_json(static_map_path)

    lane_ids = AgoMap.get_lane_ids_in_xy_bbox((min_x + max_x) / 2, (min_y + max_y) / 2, city_name, max((max_x - min_x) / 2, (max_y - min_y) / 2) + 5)
    _plot_static_map_elements(AgoMap,lane_ids,city_name)

    plt.xlim(min_x - x_buffer, max_x + x_buffer)
    plt.ylim(min_y - y_buffer, max_y + y_buffer)
    if not os.path.exists("xy9"):
        os.makedirs("xy9")
    plt.savefig(f'xy9/{scenario_id}.png')


def _plot_static_map_elements(
        AgoMap, lane_ids, city_name, show_ped_xings: bool = False
) -> None:
    ax = plt.gca()
    ax.set_facecolor(_BACKGROUND_COLOR)

    # 先画淡灰色车道线
    for lane_id in lane_ids:
        lane_boundary = AgoMap.get_lane_segment_polygon(lane_id, city_name)[:, :2]
        # plt.fill(lane_boundary[:, 0], lane_boundary[:, 1], color="#FFF59D", alpha=0.7, zorder=5)
        plt.fill(lane_boundary[:, 0], lane_boundary[:, 1], color="white", alpha=0.7, zorder=2)
        # _plot_polylines(
        #     lane_boundary,
        #     line_width=2,
        #     color=_LANE_SEGMENT_COLOR,
        # )

    # 再画黑色最左/最右边界
    #     2. 再画灰色车道线
        for lane_id in lane_ids:
            lane_boundary = AgoMap.get_lane_segment_polygon(lane_id, city_name)[:, :2]
            _plot_polylines(
                lane_boundary,
                line_width=1,
                color=_LANE_SEGMENT_COLOR, # 线层级高
            )

        # 3. 再画黑色左右边界
        # for lane_id in lane_ids:
        #     lane_boundary = AgoMap.get_lane_segment_polygon(lane_id, city_name)[:, :2]
        #     N = lane_boundary.shape[0]
        #     if N < 4:
        #         continue
        #     left_side = lane_boundary[:N // 2]
        #     right_side = lane_boundary[N - 1:N // 2 - 1:-1]
        #     plt.plot(left_side[:, 0], left_side[:, 1], color='grey', alpha=0.5,linewidth=1, zorder=101)
        #     plt.plot(right_side[:, 0], right_side[:, 1], color='grey',alpha=0.5, linewidth=1, zorder=101)

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


