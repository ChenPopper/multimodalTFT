import torch
import torch.nn as nn
from torch import _assert

from .feature_extractor import FeedForward


class MHAttention(nn.Module):
    """Customized multi-head attention for cross attention
    """

    def __init__(
            self,
            num_q_channels: int,
            num_kv_channels: int,
            num_heads: int,
            dropout: float,
            batch_first: bool = True
    ):
        super().__init__()
        _assert(
            not (num_q_channels % num_heads),
            f"num_q_channels must be divisible by num_heads, but got num_q_channels = {num_q_channels} "
            f"and num_heads = {num_heads}"
        )
        self.batch_first = batch_first  # for torch version lower than 2
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            # batch_first=True,
        )

    def forward(self, x_q, x_kv):
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)
        out = self.attention(x_q, x_kv, x_kv)[0]

        return out.transpose(0, 1)


class CrossAttention(nn.Module):
    __model_name__ = 'cross-attention'

    def __init__(
            self,
            num_q_channels: int,
            num_kv_channels: int,
            num_heads: int,
            hidden_dim: int = 32,
            num_iter: int = 3,
            dropout: float = 0.0,
            is_causal: bool = False,
            device=torch.device('cpu')
    ):
        super().__init__()
        self.is_causal = is_causal
        # self.num_iter = num_iter
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attn1 = MHAttention(
            num_q_channels=num_q_channels,
            num_kv_channels=num_kv_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc1 = FeedForward(
            input_dim=num_q_channels,
            output_dim=hidden_dim,
            hidden_dims=[2 * hidden_dim, ]
        )
        self.attn2 = MHAttention(
            num_q_channels=num_kv_channels,
            num_kv_channels=num_q_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc2 = FeedForward(
            input_dim=num_kv_channels,
            output_dim=hidden_dim,
            hidden_dims=[2 * hidden_dim, ]
        )
        self.out_q_norm = nn.LayerNorm(hidden_dim)
        self.out_kv_norm = nn.LayerNorm(hidden_dim)
        if num_iter > 1:
            rec_attn = []
            for n in range(num_iter - 1):
                rec_attn.append([
                    (
                        MHAttention(hidden_dim, hidden_dim, num_heads, dropout).to(device),
                        FeedForward(input_dim=hidden_dim, output_dim=hidden_dim, hidden_dims=[2 * hidden_dim, ]).to(
                            device)
                    ),
                    (
                        MHAttention(hidden_dim, hidden_dim, num_heads, dropout).to(device),
                        FeedForward(input_dim=hidden_dim, output_dim=hidden_dim, hidden_dims=[2 * hidden_dim, ]).to(
                            device)
                    ),
                ])
            self.rec_attn = rec_attn  # nn.Sequential(*rec_attn)
        else:
            self.rec_attn = None
        self.attn_out = MHAttention(
            num_q_channels=hidden_dim,
            num_kv_channels=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc_out = FeedForward(
            input_dim=hidden_dim,
            output_dim=num_q_channels,
            hidden_dims=[2 * hidden_dim, ]
        )

    def forward(self, x_q, x_kv):
        """
        x_q: shape = (B, C_q, E) or (B, T, C_q, E)
        x_kv: shape = (B, C_kv, E) or (B, T, C_kv, E)
        """
        if self.is_causal and (x_q.dim() == 4):
            hist_len = x_q.shape[1]
            list_h_state = []
            for i in range(hist_len):
                for j in range(i + 1):
                    query = x_q[:, i, ...]
                    kv = x_kv[:, j, ...]
                    query = self.q_norm(query)
                    kv = self.kv_norm(kv)
                    query_new = self.fc1(self.attn1(query, kv))
                    kv_new = self.fc2(self.attn2(kv, query))
                    query_new, kv_new = self.recurrent_attn(query_new, kv_new)
                    query_new = self.out_q_norm(query_new)
                    kv_new = self.out_kv_norm(kv_new)
                    list_h_state.append(self.fc_out(self.attn_out(query_new, kv_new)))
            out = torch.cat(list_h_state, dim=1)  # (B, T*(T+1)/2, E)
        elif (x_q.dim() == 4) or (x_q.dim() == 3):
            x_q = x_q.view(-1, x_q.size(-2), x_q.size(-1))
            x_kv = x_kv.view(-1, x_kv.size(-2), x_kv.size(-1))
            x_q = self.q_norm(x_q)
            x_kv = self.kv_norm(x_kv)
            x_q_new = self.fc1(self.attn1(x_q, x_kv))
            x_kv_new = self.fc2(self.attn2(x_kv, x_q))
            x_q_new, x_kv_new = self.recurrent_attn(x_q_new, x_kv_new)
            x_q_new = self.out_q_norm(x_q_new)
            x_kv_new = self.out_kv_norm(x_kv_new)
            out = self.fc_out(self.attn_out(x_q_new, x_kv_new))  # (B*T, C_q, E)
        else:
            raise RuntimeError("Unsupported input tensors!")

        return out

    def recurrent_attn(self, q, kv):
        if self.rec_attn is None:
            return q, kv
        else:
            for q_attn, kv_attn in self.rec_attn:
                q = self.out_q_norm(q_attn[1](q_attn[0](q, kv)))
                kv = self.out_kv_norm(kv_attn[1](kv_attn[0](kv, q)))
            return q, kv


# class SelfAttention(nn.Module):
#     __model_name__ = 'self-attention'

#     def __init__(self, num_channels: int, num_heads: int, dropout: float = 0.0):
#         super().__init__()
#         self.norm = nn.LayerNorm(num_channels)
#         self.attn = MHAttention(
#             num_q_channels=num_channels,
#             num_kv_channels=num_channels,
#             num_heads=num_heads,
#             dropout=dropout
#         )
#         self.out_norm = nn.LayerNorm(num_channels)

#     def forward(self, x):
#         """
#         x: shape = (B, C, E) or (B, T, C, E)
#         """
#         if x.dim() == 4:
#             hist_len = x.shape[1]
#             list_h_state = []
#             for i in range(hist_len):
#                 x_in = x[:, i, ...]
#                 x_in = self.norm(x_in)
#                 x_out = self.attn(x_in, x_in)
#                 x_out = self.out_norm(x_out)
#                 list_h_state.append(x_out)
#             return torch.cat(list_h_state, dim=1)  # (B, T, C, E)
#         elif x.dim() == 3:
#             x_in = self.norm(x)
#             x_out = self.attn(x_in, x_in)
#             x_out = self.out_norm(x_out)
#             return x_out  # (B, C, E)
#         else:
#             raise RuntimeError("Unsupported input tensors!")

class SelfAttention(nn.Module):
    __model_name__ = 'self-attention'

    def __init__(self, num_channels: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = 64
        self.input_norm = nn.LayerNorm(num_channels)
        self.attn_1 = MHAttention(
            num_q_channels=num_channels,
            num_kv_channels=num_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        self.post_attn_norm_1 = nn.LayerNorm(num_channels)
        self.fc1 = FeedForward(
            input_dim=num_channels,
            output_dim=self.hidden_dim,
            hidden_dims=[512, 128],
            dropout=dropout
        )
        self.out_norm_1 = nn.LayerNorm(self.hidden_dim)
        self.attn_2 = MHAttention(
            num_q_channels=self.hidden_dim,
            num_kv_channels=self.hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.post_attn_norm_2 = nn.LayerNorm(self.hidden_dim)
        self.fc2 = FeedForward(
            input_dim=self.hidden_dim,
            output_dim=num_channels,
            hidden_dims=[512, 128],
            dropout=dropout
        )
        self.out_norm_2 = nn.LayerNorm(num_channels)

    def forward(self, x):
        """
        x: shape = (B, C, E) or (B, T, C, E)
        """
        if x.dim() == 4:
            hist_len = x.shape[1]
            list_h_state = []
            for i in range(hist_len):
                x_in = x[:, i, ...]
                x_in = self.input_norm(x_in)
                x_out = self.post_attn_norm_1(self.attn_1(x_in, x_in) + x_in)
                x_out = self.out_norm_1(self.fc1(x_out))
                x_out = self.post_attn_norm_2((self.attn_2(x_out, x_out) + x_out))
                x_out = self.out_norm_2(self.fc2(x_out))
                list_h_state.append(x_out)
            return torch.cat(list_h_state, dim=1)  # (B, T, C, E)
        elif x.dim() == 3:
            x_in = self.input_norm(x)
            x_out = self.post_attn_norm_1(self.attn_1(x_in, x_in) + x_in)
            x_out = self.out_norm_1(self.fc1(x_out))
            x_out = self.post_attn_norm_2((self.attn_2(x_out, x_out) + x_out))
            x_out = self.out_norm_2(self.fc2(x_out))
            return x_out  # (B, C, E)
        else:
            raise RuntimeError("Unsupported input tensors!")


class MMCrossAttention(nn.Module):
    __model_name__ = 'cross-attention_mmtft'

    def __init__(
            self,
            num_q_channels: int,
            num_kv_channels: int,
            num_heads: int,
            hidden_dim: int = 32,
            num_iter: int = 3,
            dropout: float = 0.0,
            is_causal: bool = False,
            device=torch.device('cpu')
    ):
        super().__init__()
        self.is_causal = is_causal
        # self.num_iter = num_iter
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attn1 = MHAttention(
            num_q_channels=num_q_channels,
            num_kv_channels=num_kv_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc1 = FeedForward(
            input_dim=num_q_channels,
            output_dim=hidden_dim,
            hidden_dims=[2 * hidden_dim, ]
        )
        self.attn2 = MHAttention(
            num_q_channels=num_kv_channels,
            num_kv_channels=num_q_channels,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc2 = FeedForward(
            input_dim=num_kv_channels,
            output_dim=hidden_dim,
            hidden_dims=[2 * hidden_dim, ]
        )
        self.out_q_norm = nn.LayerNorm(hidden_dim)
        self.out_kv_norm = nn.LayerNorm(hidden_dim)
        if num_iter > 1:
            rec_attn = []
            for n in range(num_iter - 1):
                rec_attn.append([
                    (
                        MHAttention(hidden_dim, hidden_dim, num_heads, dropout).to(device),
                        FeedForward(input_dim=hidden_dim, output_dim=hidden_dim, hidden_dims=[2 * hidden_dim, ]).to(
                            device)
                    ),
                    (
                        MHAttention(hidden_dim, hidden_dim, num_heads, dropout).to(device),
                        FeedForward(input_dim=hidden_dim, output_dim=hidden_dim, hidden_dims=[2 * hidden_dim, ]).to(
                            device)
                    ),
                ])
            self.rec_attn = rec_attn  # nn.Sequential(*rec_attn)
        else:
            self.rec_attn = None
        self.attn_out = MHAttention(
            num_q_channels=hidden_dim,
            num_kv_channels=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.fc_out = FeedForward(
            input_dim=hidden_dim,
            output_dim=num_q_channels,
            hidden_dims=[2 * hidden_dim, ]
        )

    def forward(self, x_q, x_kv):
        """
        x_q: shape = (B, C_q, E) or (B, T, C_q, E)
        x_kv: shape = (B, C_kv, E) or (B, T, C_kv, E)
        """
        out = []
        x_q = x_q.view(-1, x_q.size(-2), x_q.size(-1))
        x_kv = x_kv.view(-1, x_kv.size(-2), x_kv.size(-1))
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        x_q_new = self.fc1(self.attn1(x_q, x_kv))
        x_kv_new = self.fc2(self.attn2(x_kv, x_q))
        x_q_new = self.out_q_norm(x_q_new)
        x_kv_new = self.out_kv_norm(x_kv_new)
        out.append(self.fc_out(self.attn_out(x_q_new, x_kv_new)))
        out.extend(self.recurrent_attn(x_q_new, x_kv_new))
        # x_q_new = self.out_q_norm(x_q_new)
        # x_kv_new = self.out_kv_norm(x_kv_new)
        # out = self.fc_out(self.attn_out(x_q_new, x_kv_new))  # (B*T, C_q, E)

        return out

    def recurrent_attn(self, q, kv):
        if self.rec_attn is None:
            return []
        else:
            out = []
            for q_attn, kv_attn in self.rec_attn:
                q = self.out_q_norm(q_attn[1](q_attn[0](q, kv)))
                kv = self.out_kv_norm(kv_attn[1](kv_attn[0](kv, q)))
                x_q_new = self.out_q_norm(q)
                x_kv_new = self.out_kv_norm(kv)
                out.append(self.fc_out(self.attn_out(x_q_new, x_kv_new)))
            return out

# if __name__ == '__main__':
#     # model = SelfAttention(1, 1)
#     model = CrossAttention(1, 1, 1, )
#     print(model.__model_name__)
