import os.path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
import math

from argoverse.evaluation.competition_util import generate_forecasting_h5

from losses import Huber2DLoss
from losses import CELoss
from metrics import BrierMinFDE
from metrics import MR
from metrics import MinADE
from metrics import MinFDE
from modules import Backbone_real2Shared8MoE_CoPRbias_QLinear64_LasttripleLayer as Backbone
from modules import MapEncoder
from visualization import trajectory_visualization
import pickle
from utils import generate_target
from utils import generate_predict_mask
from utils import AlignWithHistory
import numpy as np
import gc

torch.set_float32_matmul_precision('high')

def switch_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    router_probs = torch.softmax(router_logits, dim=-1)  # [b*s, num_experts]
    _, selected_experts = torch.topk(router_probs, k=1, dim=-1)  # [b*s, 1]
    selected_experts = selected_experts.squeeze(-1)  # [b*s]
    mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()  # [b*s, num_experts]
    actual_load = mask.mean(dim=0)  # [num_experts]
    expected_load = 1.0 / num_experts
    aux_loss = torch.mean((actual_load - expected_load) ** 2) * num_experts
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001  # 可调整的超参数
    total_loss = aux_loss + z_loss * z_loss_weight
    return total_loss

def bug_switch_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    计算 Switch Transformers 的负载均衡损失

    Args:
        router_logits: shape [batch_size * sequence_length, num_experts]
        num_experts: 专家数量

    Returns:
        total_loss: 总损失 = auxiliary_loss + z_loss
    """
    # 计算路由概率
    router_probs = torch.softmax(router_logits, dim=-1)  # [b*s, num_experts]

    # 获取每个token的最优专家
    _, selected_experts = torch.topk(router_probs, k=2, dim=-1)  # [b*s]

    # 创建one-hot矩阵表示选中的专家
    mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()  # [b*s, num_experts]

    # 计算每个专家的期望负载 (理想情况下应该是 1/num_experts)
    expected_load = torch.ones_like(router_probs) / num_experts

    # 计算实际负载 (每个专家处理的token数量除以总token数量)
    # 在batch维度上计算平均值
    actual_load = mask.mean(dim=0)  # [num_experts]

    # 计算auxiliary loss
    # 这会惩罚负载分布与期望负载的差异
    aux_loss = torch.sum(actual_load * router_probs.mean(dim=0)) * num_experts

    # 计算z_loss (可选)
    # 这会惩罚过大的路由logits
    z_loss = torch.mean(torch.square(router_logits))
    z_loss_weight = 0.001  # 可调整的超参数

    # 总损失
    total_loss = aux_loss + z_loss * z_loss_weight

    return total_loss


class InMoE_real2Shared8MoE_CoPRbias_stepLR_QLinear64_Lasttriplelayer(pl.LightningModule):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 pos_duration: int,
                 pred_duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_visible_steps: int,
                 num_modes: int,
                 num_attn_layers: int,
                 num_hops: int,
                 num_heads: int,
                 dropout: float,
                 lr: float,
                 weight_decay: float,
                 warmup_epochs: int,
                 T_max: int,
                 is_training: bool,
                 **kwargs) -> None:
        super(InMoE_real2Shared8MoE_CoPRbias_stepLR_QLinear64_Lasttriplelayer, self).__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.pos_duration = pos_duration
        self.pred_duration = pred_duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_visible_steps = num_visible_steps
        self.num_modes = num_modes
        self.num_attn_layers = num_attn_layers
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.is_training = is_training

        self.Backbone = Backbone(
            hidden_dim=hidden_dim,
            num_historical_steps=num_historical_steps,
            num_future_steps=num_future_steps,
            pos_duration=pos_duration,
            pred_duration=pred_duration,
            a2a_radius=a2a_radius,
            l2a_radius=l2a_radius,
            num_attn_layers=num_attn_layers,
            num_modes=num_modes,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.MapEncoder = MapEncoder(
            hidden_dim=hidden_dim,
            num_hops=num_hops,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reg_loss = Huber2DLoss()
        self.prob_loss = CELoss()

        self.brier_minFDE = BrierMinFDE()
        self.minADE = MinADE()
        self.minFDE = MinFDE()
        self.MR = MR()

        self.test_traj_output = dict()
        self.test_prob_output = dict()
        self.gen_split = 'gen_vis'
        self.planning_dict = dict()
        self.vis_result = dict()

    def forward(self,
                data: Batch):
        lane_embs = self.MapEncoder(data=data)
        pred = self.Backbone(data=data, l_embs=lane_embs)
        return pred

    def training_step(self, data, batch_idx):
        traj_propose, traj_output, prob_output, router_logits_list = self(
            data)  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]
        target_traj, target_mask = generate_target(position=data['agent']['position'],
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  # [(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        errors = (torch.norm(traj_propose[..., :2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(
            2)).sum(dim=-1)  # [(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)  # [(N1,...Nb),H]
        traj_best_propose = traj_propose[
            torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None,
                                                         :], best_mode_index]  # [(N1,...Nb),H,F,2]
        traj_best_output = traj_output[
            torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None,
                                                        :], best_mode_index]  # [(N1,...Nb),H,F,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:, :self.num_historical_steps],
                                             self.num_visible_steps)  # [(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]  # [Na,F]
        traj_pro = traj_best_propose[predict_mask]  # [Na,F,2]
        traj_ref = traj_best_output[predict_mask]  # [Na,F,2]
        prob = prob_output[predict_mask]  # [Na,K]
        targ = target_traj[predict_mask]  # [Na,F,2]
        label = best_mode_index[predict_mask]  # [Na]

        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask])
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])
        prob_loss = self.prob_loss(prob, label)

        router_loss = 0
        for router_logits in router_logits_list:
            router_loss += switch_load_balancing_loss(router_logits, 8) # 8个专家
        alpha = 0.01
        loss = reg_loss_propose + reg_loss_refine + prob_loss + alpha * router_loss
        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=True, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('train_prob_loss', prob_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1, sync_dist=True)

        return loss

    def validation_step(self, data, batch_idx):
        traj_propose, traj_output, prob_output, _ = self(
            data)  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]
        target_traj, target_mask = generate_target(position=data['agent']['position'],
                                                   mask=data['agent']['visible_mask'],
                                                   num_historical_steps=self.num_historical_steps,
                                                   num_future_steps=self.num_future_steps)  # [(N1,...Nb),H,F,2],[(N1,...Nb),H,F]

        errors = (torch.norm(traj_propose[..., :2] - target_traj.unsqueeze(2), p=2, dim=-1) * target_mask.unsqueeze(
            2)).sum(dim=-1)  # [(N1,...Nb),H,K]
        best_mode_index = errors.argmin(dim=-1)  # [(N1,...Nb),H]
        traj_best_propose = traj_propose[
            torch.arange(traj_propose.size(0))[:, None], torch.arange(traj_propose.size(1))[None,
                                                         :], best_mode_index]  # [(N1,...Nb),H,F,2]
        traj_best_output = traj_output[
            torch.arange(traj_output.size(0))[:, None], torch.arange(traj_output.size(1))[None,
                                                        :], best_mode_index]  # [(N1,...Nb),H,F,2]

        predict_mask = generate_predict_mask(data['agent']['visible_mask'][:, :self.num_historical_steps],
                                             self.num_visible_steps)  # [(N1,...Nb),H]
        targ_mask = target_mask[predict_mask]  # [Na,F]
        traj_pro = traj_best_propose[predict_mask]  # [Na,F,2]
        traj_ref = traj_best_output[predict_mask]  # [Na,F,2]
        prob = prob_output[predict_mask]  # [Na,K]
        targ = target_traj[predict_mask]  # [Na,F,2]
        label = best_mode_index[predict_mask]  # [Na]

        reg_loss_propose = self.reg_loss(traj_pro[targ_mask], targ[targ_mask])
        reg_loss_refine = self.reg_loss(traj_ref[targ_mask], targ[targ_mask])
        prob_loss = self.prob_loss(prob, label)
        loss = reg_loss_propose + reg_loss_refine + prob_loss
        self.log('val_reg_loss_propose', reg_loss_propose, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_reg_loss_refine', reg_loss_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)
        self.log('val_prob_loss', prob_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
        num_agents = agent_index.size(0)
        agent_traj = traj_output[agent_index, -1]  # [N,K,F,2]
        agent_prob = prob_output[agent_index, -1]  # [N,K]
        agent_targ = target_traj[agent_index, -1]  # [N,F,2]
        fde = torch.norm(agent_traj[:, :, -1, :2] - agent_targ[:, -1, :2].unsqueeze(1), p=2, dim=-1)  # [N,K]
        best_mode_index = fde.argmin(dim=-1)  # [N]
        agent_traj_best = agent_traj[torch.arange(num_agents), best_mode_index]  # [N,F,2]
        self.brier_minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2],
                                 agent_prob[torch.arange(num_agents), best_mode_index])
        self.minADE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.minFDE.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.MR.update(agent_traj_best[..., :2], agent_targ[..., :2])
        self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents,
                 sync_dist=True)
        self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents,
                 sync_dist=True)
        self.log('val_MR', self.MR, prog_bar=True, on_step=False, on_epoch=True, batch_size=num_agents, sync_dist=True)
        self.log('val_brier_minFDE', self.brier_minFDE, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=num_agents, sync_dist=True)

        # vis
        # trajectory_visualization(data, traj_output, is_test=False)

    def test_step(self, data, batch_idx):
        traj_propose, traj_output, prob_output,_ = self(
            data)  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K]

        prob_output = prob_output ** 2
        prob_output = prob_output / prob_output.sum(dim=-1, keepdim=True)

        if 'gen_vis' in self.gen_split:
            agent_index = data['agent']['agent_index'] + data['agent']['ptr'][:-1]
            num_agents = agent_index.size(0)
            agent_traj = traj_output[agent_index, -1]  # [N,K,F,2]
            agent_prob = prob_output[agent_index, -1]  # [N,K]
            all_gt_traj = data['agent']['position'][agent_index] # [N,F,2]
            for i in range(num_agents):
                gt_histfuture_traj = all_gt_traj[i].cpu().numpy()
                history_accumulate_length = np.sum(np.sqrt(np.sum(np.diff(gt_histfuture_traj[:20], axis=0) ** 2, axis=1)))

                intention,angle_variation = AlignWithHistory(gt_histfuture_traj,20)

                id = int(data['scenario_id'][i])
                traj = agent_traj[i].cpu().numpy()
                prob = agent_prob[i].tolist()

                traj_end = traj[:, -1, :]  # [6, 2]
                # 真实轨迹的最后一个点
                gt_end = gt_histfuture_traj[-1, :]  # [2]
                # 计算每个预测轨迹终点与真实终点的欧式距离
                fde = np.linalg.norm(traj_end - gt_end, axis=1)  # [6]
                min_fde = np.min(fde)

                # self.test_traj_output[id] = traj
                # self.test_prob_output[id] = prob
                # self.minFDE_output[id] = min_fde
                self.vis_result[id] = dict()
                self.vis_result[id]['traj'] = traj
                self.vis_result[id]['prob'] = prob
                self.vis_result[id]['minFDE'] = min_fde

        else:
            # planning_dict = dict()
            agent_traj = traj_output[:, -1]
            agent_prob = prob_output[:, -1]
            scenario_id = data['scenario_id']
            for scenario_count in range(len(scenario_id)):
                start_index = data['agent']['ptr'][scenario_count].item()
                stop_index = data['agent']['ptr'][scenario_count+1].item()
                self.planning_dict[scenario_id[scenario_count]] = dict()
                self.planning_dict[scenario_id[scenario_count]]['traj'] = agent_traj[start_index:stop_index].cpu().numpy()
                self.planning_dict[scenario_id[scenario_count]]['prob'] = agent_prob[start_index:stop_index].cpu().numpy()
            # for i in range(agent_traj.size(0)):
            #     id = int(scenario_id[i])
            #     traj = agent_traj[i].cpu().numpy()
            #     prob = agent_prob[i].tolist()
            #
            #     if id not in planning_dict:
            #         planning_dict[id] = dict()
            #     planning_dict[id]['traj'] = traj
            #     planning_dict[id]['prob'] = prob

            # vis
        # trajectory_visualization(data, traj_output, is_test=True)
    #def on_train_batch_end(self, outputs, batch, batch_idx):
                # 如果只想在训练 step 后清理，可以在这里执行
    #    torch.cuda.empty_cache()
    #    gc.collect()

    def on_test_end(self):
        if self.gen_split == 'Planning_output':
            output_path = './RL_plan_output'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filename = self.gen_split + '.submission'
            with open(os.path.join(output_path, filename), 'wb') as f:
                pickle.dump(self.planning_dict, f)
        elif 'gen_vis' in self.gen_split:
            output_path = './vis_output'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filename = self.gen_split + '.submission'
            with open(os.path.join(output_path, filename), 'wb') as f:
                pickle.dump(self.vis_result, f)
        else:
            output_path = './RL_plan_output'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            filename = self.gen_split+'.submission'
            generate_forecasting_h5(self.test_traj_output, output_path, filename, self.test_prob_output)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        # watch_else_list = []
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)

                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)

        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} are in both decay and no_decay sets!"
        assert len(
            param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} are not in either decay or no_decay sets!"
        # watch_else_list = param_dict.keys() - union_params
        # for param_name in watch_else_list:
        #     decay.add(param_name)
        #     # no_decay.remove(param_name)

        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": self.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr, weight_decay=self.weight_decay)

        warmup_epochs = self.warmup_epochs

        def warmup_cosine_annealing_schedule(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            else:
                if epoch < 26:
                    return 1.0
                else:
                    cifang = (epoch - 26) // 10 + 1
                    return 0.5**cifang
            # return 1.0

        # milestones = [4+20,4+30,4+40,4+50]
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_annealing_schedule),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('InMoE_real2Shared8MoE_CoPRbias_stepLR_QLinear64_Lasttriplelayer')
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--num_historical_steps', type=int, default=20)
        parser.add_argument('--num_future_steps', type=int, default=30)
        parser.add_argument('--pos_duration', type=int, default=20)
        parser.add_argument('--pred_duration', type=int, default=20)
        parser.add_argument('--a2a_radius', type=float, default=50)
        parser.add_argument('--l2a_radius', type=float, default=50)
        parser.add_argument('--num_visible_steps', type=int, default=2)
        parser.add_argument('--num_modes', type=int, default=6)
        parser.add_argument('--num_attn_layers', type=int, default=2)
        parser.add_argument('--num_hops', type=int, default=4)
        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=3.5e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--warmup_epochs', type=int, default=4)
        parser.add_argument('--T_max', type=int, default=72)
        return parent_parser


# if __name__== "__main__":
