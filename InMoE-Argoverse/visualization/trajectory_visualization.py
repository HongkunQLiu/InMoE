import matplotlib.pyplot as plt
from argoverse.map_representation.map_api import ArgoverseMap
import numpy as np
import torch
import os
from torch_geometric.data import Batch
from torch_geometric.utils import unbatch

from visualization.trajectory_visualization5 import plot_single_vehicle,plot_obj_pose,plot_obj_pose2

num_historical_steps = 20
situation = 'SIMPL'#'RARE'#'SIMPL'
def trajectory_visualization(save_dir,map_api,data:Batch, traj_output: torch.tensor, is_test: torch.bool=False) -> None:
    batch_size = len(data['scenario_id'])
    city_name = data['city']

    agent_batch = data['agent']['batch']
    agent_position = data['agent']['position'].detach()
    agent_position = unbatch(agent_position, agent_batch)
    num_modes = traj_output.size(2)
    traj_output = traj_output.detach()
    traj_output = unbatch(traj_output[:,-1], agent_batch)
    agent_index = data['agent']['agent_index']


    
    for i in range(batch_size):
        scenario_id = data['scenario_id'][i]
        agent_position_i = agent_position[i][agent_index[i]].squeeze(0)
        agent_historical_position = agent_position_i[:num_historical_steps].cpu().numpy()
        agent_future_position = agent_position_i[num_historical_steps:].cpu().numpy()
        agent_prediction_position = traj_output[i][agent_index[i]].squeeze(0).cpu().numpy()
        plt.figure()
        if 'SIMPL' in situation:
            result_pt_dir = "/data2/lhk/SIMPL"
            agent_prediction_position_dict = torch.load(os.path.join(result_pt_dir, f"reglist_{scenario_id}.pt"))
            agent_prediction_position = agent_prediction_position_dict['reg_result']
            agent_prediction_position = torch.stack(agent_prediction_position).numpy()
        if 'RARE' in situation:
            result_pt_dir = "/data2/lhk/InMoE_analyze/rare_scenario_features"
            save_feature = torch.load(os.path.join(result_pt_dir, f"MoEIntentionRelationship_{scenario_id}.pt"))
            history_length_list = save_feature['history_length_list'][0]
            intention_list = save_feature['intention_list'][0]
            angle_variation_list = save_feature['angle_variation_list'][0]
            # agent_prediction_position = agent_prediction_position_dict['reg_result']
            # agent_prediction_position = torch.stack(agent_prediction_position).numpy()
        plot_single_vehicle(agent_historical_position,  # 过去轨迹的坐标数组，形状为 (1, 50, 2)
        agent_future_position, # 地面真实轨迹的坐标数组，形状为 (1, 60, 2)
        agent_prediction_position,  # 预测轨迹的列表，每个数组形状为 (6,60, 2)
        data["scenario_id"][i],
        map_api,city_name[i])

        all_agent_position = agent_position[i].cpu().numpy()
        for other_cnt in range(len(agent_position[i])):
            if other_cnt == agent_index[i]:
                continue
            agent_position_trajectory = all_agent_position[other_cnt][:num_historical_steps]
            # plt.plot(agent_position_trajectory[-1, 0], agent_position_trajectory[-1, 1], markersize=8, marker='o',
            #          color='blue', zorder=100,
            #          alpha=0.5)
            pt1 = agent_position_trajectory[-2]
            pt2 = agent_position_trajectory[-1]
            center_x, center_y = pt2[0], pt2[1]
            center_z = 0
            length, width, height = 6.0, 1.7, 1.5
            heading = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
            velocity_x, velocity_y = pt2[0] - pt1[0], pt2[1] - pt1[1]
            state = np.array([center_x, center_y, center_z, length, width, height, heading, velocity_x, velocity_y, 1])
            plot_obj_pose2('TYPE_VEHICLE', state, ax=plt.gca(),alpha=0.7)

        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        save_dir = "SIMPL_0711"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
         
        plt.savefig(os.path.join(save_dir,f'SIMPL_0711_4_{data["scenario_id"][i]}.png'),dpi=300)
        print(f'saved {data["scenario_id"][i]}')
        #plt.savefig(os.path.join(save_dir,scenario_id+f'_{history_length_list}_{angle_variation_list}_{intention_list}.png'), dpi=300)
        # if is_test:
        #     os.makedirs('test_output/visualization', exist_ok=True)
        #     plt.savefig(f'test_output/visualization/{data["scenario_id"][i]}.png')
        # else:
        #     os.makedirs('visualization/trajectory', exist_ok=True)
        #     plt.savefig(f'visualization/trajectory/{data["scenario_id"][i]}.png')
        # plt.close()
