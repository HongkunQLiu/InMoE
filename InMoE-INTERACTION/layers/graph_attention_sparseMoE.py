from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import torch.nn.functional as F
import argparse
from transformers.activations import ACT2FN

from utils import init_weights


class NaiveBasicExpert(nn.Module):
    # 一个 Expert 可以是一个最简单的， linear 层即可
    # 也可以是 MLP 层
    # 也可以是 更复杂的 MLP 层（active function 设置为 swiglu）
    def __init__(self, feature_in, feature_out):
        super().__init__()
        self.linear = nn.Linear(feature_in, feature_out)

    def forward(self, x):
        return self.linear(x)


class BasicExpert(nn.Module):
    def __init__(self, hidden_dim, dropout):   # LLM 进化之路， FFN 激活函数从 GELU -> SwishGLU
        super().__init__()

        # 有一个 magic number 叫做 8/3
        hidden_dim = hidden_dim
        # 这里可以自己去优化成 multiple_of 的倍数
        mid_dim = hidden_dim * 8 // 3

        self.up = nn.Linear(hidden_dim, mid_dim, bias=False)
        self.down = nn.Linear(mid_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, mid_dim, bias=False)

        self.dropout = nn.Dropout()

    def forward(self, x):
        out = self.dropout(
            self.down(
                # up 之后的 Shape 是(b, s, mid_dim)
                # gate 和 up 之后的Shape都是 (b, s, mid_dim)
                # 两者是 element-wise 相乘
                F.silu(self.gate(x)) * self.up(x)
            )
        )
        return out

class MOEConfig:
    def __init__(
            self,
            hidden_dim,
            expert_number,
            top_k,
            shared_experts_number=0,
        ):
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number


class MOERouter(nn.Module):
    def __init__(self, hidden_dim, expert_number, top_k):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, expert_number)
        self.expert_number = expert_number
        self.top_k = top_k

    def forward(self, hidden_states):
        # 计算路由logits
        router_logits = self.gate(hidden_states)  # shape is (b * s, expert_number)

        # 计算专家经过softmax之后的概率
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)

        # 计算topk的专家的输出
        router_weights, selected_experts = torch.topk(
            routing_probs, self.top_k, dim=-1
        )  # shape都是 (b * s, top_k)

        # 专家权重归一化
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True)
        router_weights = router_weights.to(hidden_states.dtype)

        # 生成专家掩码
        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.expert_number
        )  # shape是 (b * s, top_k, expert_number)
        expert_mask = expert_mask.permute(2, 1, 0)  # (expert_number, top_k, b * s)

        return router_logits, router_weights, selected_experts, expert_mask

class SparseMOE(nn.Module):
    # 稀疏 MOE 模型，这里每一个 token 都会过 topk 个专家，得到对应token 的 hidden_embeddings
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.expert_number = config.expert_number
        self.top_k = config.top_k

        self.experts = nn.ModuleList(
            [
                BasicExpert(self.hidden_dim, self.hidden_dim) for _ in range(self.expert_number)
            ]
        )

        self.router = MOERouter(self.hidden_dim, self.expert_number, self.top_k)

    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size_multi_seq_len, hidden_dim = x.size()

        # 合并前两个维度，因为不是 Sample 维度了，而是 token 维度
        hidden_states = x.view(-1, hidden_dim)  # shape is(b * s, hidden_dim)

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(hidden_states)
        # 其中 selected_experts_indices shape 是 (b * s, top_k)
        # 其中 expert_mask shape 是 (expert_number, top_k, b * s)

        final_hidden_states = torch.zeros(
            (batch_size_multi_seq_len, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx] shape 是 (top_k, b * s)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx 和 top_x 都是一维 tensor
            # idx 的值是 0 或 1, 表示这个 token 是作为当前专家的 top1 还是 top2
            # top_x 的值是 token 在 batch*seq_len 中的位置索引
            # 例如对于 batch_size=2, seq_len=4 的输入:
            # top_x 的值范围是 0-7, 表示在展平后的 8 个 token 中的位置
            # idx 的值是 0/1, 表示这个 token 把当前专家作为其 top1/top2 专家

            # hidden_states 的 shape 是 (b * s, hidden_dim)
            # 需要取到 top_x 对应的 hidden_states
            current_state = hidden_states.unsqueeze(
                0
            )[:, top_x, :].reshape(-1, hidden_dim)  # （selected_token_number, hidden_dim）

            # router_weight 的 shape 是 (b * s, top_k)
            current_hidden_states = expert_layer(
                current_state
            ) * router_weights[top_x, idx].unsqueeze(-1)  # （selected_token_number, 1） 这里有广播

            # 把当前专家的输出加到 final_hidden_states 中
            # 方式1 的写法性能更好，并且方式1容易出现
            # final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            # 方式2
            final_hidden_states[top_x] += current_hidden_states.to(hidden_states.dtype)
            # 方式1 的写法性能更差，并且方式1容易出现错误，+= 操作在处理重复索引时需要多次读写内存，可能会导致竞争条件

        # 把 final_hidden_states 还原到原来的 shape
        final_hidden_states = final_hidden_states.reshape(batch_size_multi_seq_len, hidden_dim)

        return final_hidden_states , router_logits  # shape 是 (b * s, expert_number)


class ShareExpertMOE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.moe_model = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(
                    config.hidden_dim, config.hidden_dim
                ) for _ in range(config.shared_experts_number)
            ]
        )
        self.shared_experts_number = config.shared_experts_number

    def forward(self, x):
        # x shape 是 (b, s, hidden_dim)
        # 首先过 moe 模型
        sparse_moe_out, router_logits = self.moe_model(x)

        # 针对的还是 x 的每一个
        # 然后过 shared experts
        if self.shared_experts_number == 0:
            #print("without shared experts")
            return sparse_moe_out, router_logits
        else:
            shared_experts_out = [
                expert(x) for expert in self.shared_experts
            ]  # 每一个 expert 的输出 shape 是 (b, s, hidden_dim)

            shared_experts_out = torch.stack(
                shared_experts_out, dim=0
            ).sum(dim=0)

            # 把 sparse_moe_out 和 shared_experts_out 加起来
            return sparse_moe_out + shared_experts_out, router_logits

class GraphAttentionSharedlessMoE(MessagePassing):
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 has_edge_attr: bool,
                 if_self_attention: bool,
                 layer_idx: int,
                 **kwargs) -> None:
        super(GraphAttentionSharedlessMoE, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.MoEConfig_out = {
            "n_routed_experts": 4, #64
            "num_experts_per_tok": 1,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "moe_intermediate_size": 1408,
            "n_shared_experts": 2,
            "scoring_func": "softmax",
            "aux_loss_alpha": 0.001,
            "seq_aux" : True,
            "norm_topk_prob" : False,
            "hidden_size": 128, #4096
            "intermediate_size": 2736,  #10944
            "hidden_act": "silu",
            "pretraining_tp": 1
        }
        self.config = argparse.Namespace(**self.MoEConfig_out)
        self.ShareMoE_config = MOEConfig(128, 4, 1,0)
        # Linear_module = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim * 4, hidden_dim),
        # )
        # self.ffn = SparseMOE(self.ShareMoE_config) if (self.config.n_routed_experts is not None and  \
        #                                    layer_idx >= self.config.first_k_dense_replace and layer_idx % self.config.moe_layer_freq == 0) \
        #                                 else Linear_module
        self.ffn = ShareExpertMOE(self.ShareMoE_config)
        self.attn_drop = nn.Dropout(dropout)
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)
        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr)
        moe_emb, router_logits = self._ffn_layer(self.ffn_prenorm(x_dst))
        x_dst = x_dst + moe_emb
        return x_dst, router_logits

    def message(self,
                x_dst_i: torch.Tensor,
                x_src_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim)
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)
        scale = self.head_dim ** 0.5
        weight = (query_i * key_j).sum(dim=-1) / scale
        weight = softmax(weight, index, ptr)
        weight = self.attn_drop(weight)
        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads*self.head_dim)

    def _mha_layer(self,
                   x_src: torch.Tensor,
                   x_dst: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: Optional[torch.Tensor]=None) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src)

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
