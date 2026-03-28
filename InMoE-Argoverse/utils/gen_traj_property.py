import numpy as np
from scipy.spatial.distance import euclidean
import math
from sklearn.linear_model import LinearRegression

def radians_to_degrees(radians):
    return radians * 180 / math.pi

def get_heading_angle(traj):
    """
        get the heading angle
        traj: [N,2] N>=6
    """
    # length == 6
    # sort position
    _traj = traj.copy()
    traj = traj.copy()

    traj = traj[traj[:, 0].argsort()]
    traj = traj[traj[:, 1].argsort()]

    if len(traj) == 0 and np.sum(traj) == 0:
        return None

    try:
        if traj.T[0].max() - traj.T[0].min() > traj.T[1].max() - traj.T[
            1].min():  # * dominated by x #如果x方向的位移大于y方向的位移
            reg = LinearRegression().fit(traj[:, 0].reshape(-1, 1), traj[:, 1])
            traj_dir = _traj[-2:].mean(0) - _traj[:2].mean(0)
            reg_dir = np.array([1, reg.coef_[0]])
            angle = np.arctan(reg.coef_[0])
        else:
            # using y as sample and x as the target to fit a line  #如果y方向的位移大于x方向的位移
            reg = LinearRegression().fit(traj[:, 1].reshape(-1, 1), traj[:, 0])
            traj_dir = _traj[-2:].mean(0) - _traj[:2].mean(0)
            reg_dir = np.array([reg.coef_[0], 1]) * np.sign(reg.coef_[0])
            # print("reg.coef_[0]:",reg.coef_[0])
            if 1e-10 + reg.coef_[0] == 0:
                import pdb
                pdb.set_trace()
            angle = np.arctan(1 / (1e-10 + reg.coef_[0]))
    except:
        angle = 0.0
        reg_dir = 0.0
        traj_dir = 0.0
        # 注意：这里需要改
        # import pdb
        # pdb.set_trace()

    if angle < 0:
        angle = 2 * np.pi + angle
    try:
        if (reg_dir * traj_dir).sum() < 0:  # not same direction
            angle = (angle + np.pi) % (2 * np.pi)
    except:
        angle = 0.0
    # try,except 需要改
    # angle from y
    angle_to_y = angle - np.pi / 2
    angle_to_y = -angle_to_y
    '''
    x轴角度正值，
    + + + +
    ---------------->x
    _ _ _ _

    y轴角度正值
     ^    
    -｜+
    -｜+
    -｜+
    -｜+



    '''

    return angle_to_y


def convert_angle(angle):
    """
    将角度从-360到360范围内转换为-180到180范围内

    :param angle: float，需要转换的角度值
    :return: float，转换后的角度值
    """
    if angle > 180:
        angle = angle - 360
    elif angle < -180:
        angle = angle + 360
    return angle

def AlignWithHistory(trajectory,history_timestep):
    straight_angle_range = 10  # np.pi/(6*2)
    turn_angle_range = 95  # np.pi/2
    # back_angle_range = 180  # np.pi
    history_accumulate_length = np.sum(
        np.sqrt(np.sum(np.diff(trajectory[:history_timestep, :], axis=0) ** 2, axis=1)))
    history_euclidean_length = euclidean(trajectory[:history_timestep, :][0],
                                         trajectory[:history_timestep, :][-1])
    while (history_accumulate_length < 1.0 and history_euclidean_length < 1.0 and history_timestep < 70):
        history_timestep += 5
        history_accumulate_length = np.sum(
            np.sqrt(np.sum(np.diff(trajectory[:history_timestep, :], axis=0) ** 2, axis=1)))
        history_euclidean_length = euclidean(trajectory[:history_timestep, :][0],
                                             trajectory[:history_timestep, :][-1])

    history_direction_theta = get_heading_angle(trajectory[:history_timestep, :])

    future_direction_theta = get_heading_angle(trajectory[history_timestep:, :])

    if future_direction_theta is None:
        return "stay in place", 0.0
    if history_direction_theta is None:
        history_direction_theta = 0.0

    theta_delta = future_direction_theta - history_direction_theta
    theta_delta_origin = radians_to_degrees(theta_delta)

    theta_delta_angle = convert_angle(theta_delta_origin)

    # print("history_direction:",history_direction_theta_angle,'future direction:',future_direction_theta_angle,"delta angle:",theta_delta_origin,"converted angle:",theta_delta_angle)
    if theta_delta_angle < straight_angle_range and theta_delta_angle > -straight_angle_range:
        return "R1 or R2", theta_delta_angle
    elif theta_delta_angle > straight_angle_range and theta_delta_angle < turn_angle_range:
        return "R3 or R4", theta_delta_angle
    elif theta_delta_angle < -straight_angle_range and theta_delta_angle > -turn_angle_range:
        return "R5 or R6", theta_delta_angle
    else:
        return "straight backward or U-turn", theta_delta_angle