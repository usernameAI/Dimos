import numpy as np
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
import torch.nn.functional as F
from mamba_ssm import Mamba
from abc import ABC, abstractmethod
import torch.distributed as dist
import math


class fusion_triple_feature(nn.Module):
    def __init__(self, emb_dim):
        super(fusion_triple_feature, self).__init__()
        self.emb_dim = emb_dim
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.linear_final = nn.Linear(self.emb_dim, self.emb_dim)

    def forward(self, seq_hidden, pos_emb, feature_emb):
        seq_hidden = seq_hidden.unsqueeze(dim=1)
        pos_emb = pos_emb.unsqueeze(dim=1)
        feature_emb = feature_emb.unsqueeze(dim=1)
        seq_hidden = self.linear1(seq_hidden)
        pos_emb = self.linear2(pos_emb)
        feature_emb = self.linear3(feature_emb)
        fusion_feature = torch.cat((seq_hidden, pos_emb, feature_emb), dim=1)
        attn_weight = self.softmax(fusion_feature)
        fusion_feature = torch.sum(attn_weight * fusion_feature, dim=1)
        fusion_feature = self.linear_final(fusion_feature)
        return fusion_feature


class seq_affinity_soft_attention(nn.Module):
    def __init__(self, emb_dim):
        super(seq_affinity_soft_attention, self).__init__()
        self.emb_dim = emb_dim
        self.linear_1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_2 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_3 = nn.Linear(self.emb_dim, self.emb_dim)
        self.linear_alpha = nn.Linear(self.emb_dim, 1, bias=False)
        self.long_middle_short = fusion_triple_feature(self.emb_dim)

    def forward(self, mask, short_feature, seq_feature, affinity_feature):
        q1 = self.linear_1(seq_feature)
        q2 = self.linear_2(short_feature)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        q3 = self.linear_3(affinity_feature)
        q3_expand = q3.unsqueeze(1).expand_as(q1)
        alpha = self.linear_alpha(mask * torch.sigmoid(q1 + q2_expand + q3_expand))
        long_feature = torch.sum(alpha.expand_as(seq_feature) * seq_feature, 1)
        seq_output = self.long_middle_short(long_feature, short_feature, affinity_feature)
        return seq_output


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim
        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], axis=0), self.fouriercoeffs)
        y = y.view(outshape)
        return y


class Bi_Mamba_block(nn.Module):
    def __init__(self, emb_dim, d_state, d_conv, expand, drop_ratio, gridsize, shared_parameter: bool = True):
        super(Bi_Mamba_block, self).__init__()
        self.emb_dim = emb_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.drop_ratio = drop_ratio
        self.gridsize = gridsize

        self.forward_mamba_block = Mamba(
            d_model=self.emb_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        self.backward_mamba_block = Mamba(
            d_model=self.emb_dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
        )
        self.FourierKAN = NaiveFourierKANLayer(self.emb_dim, self.emb_dim, self.gridsize)
        self.ln1 = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.ln = nn.LayerNorm(self.emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=self.drop_ratio)
        if shared_parameter:
            self.backward_mamba_block.in_proj.weight = self.forward_mamba_block.in_proj.weight
            self.backward_mamba_block.in_proj.bias = self.forward_mamba_block.in_proj.bias
            self.backward_mamba_block.out_proj.weight = self.forward_mamba_block.out_proj.weight
            self.backward_mamba_block.out_proj.bias = self.forward_mamba_block.out_proj.bias

    def forward(self, x):
        out = self.ln1(x)
        out_forward = self.forward_mamba_block(out)
        out_backward = self.backward_mamba_block(out.flip(dims=(1,)))
        out_backward = out_backward.flip(dims=(1,))
        out = out_forward + out_backward
        out = self.ln(out)
        out = self.dropout(out)
        out = self.FourierKAN(out)
        out = out + x
        return out


class Bi_Mamba(nn.Module):
    def __init__(self, emb_dim, layer_num, d_state, d_conv, expand, drop_prob, gridsize):
        super(Bi_Mamba, self).__init__()
        self.emb_dim = emb_dim
        self.layer_num = layer_num
        self.drop_prob = drop_prob
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.gridsize = gridsize

        self.mamba_list = nn.ModuleList(
            Bi_Mamba_block(
                emb_dim=self.emb_dim,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                drop_ratio=self.drop_prob,
                gridsize=self.gridsize
            ) for _ in range(self.layer_num))

    def forward(self, x):
        feature_list = []
        for layer in range(self.layer_num):
            x = self.mamba_list[layer](x)
            feature_list.append(x)
        x = sum(feature_list) / len(feature_list)
        return x


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps,
                                   lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2, )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2, )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  # scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10, dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    betas.append(min(1 - alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps - 1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size * 4)
        self.w_2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (
                1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in
                   zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1, 1, 1, corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, num_blocks):
        super(Transformer_rep, self).__init__()
        self.hidden_size = hidden_size
        self.heads = n_heads
        self.dropout = dropout
        self.n_blocks = num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden


class Diffu_xstart(nn.Module):
    def __init__(self, hidden_size, dropout, num_blocks, lambda_uncertainty, d_state, d_conv, expand, gridsize):
        super(Diffu_xstart, self).__init__()
        self.hidden_size = hidden_size
        # self.heads = n_heads
        self.dropout = dropout
        self.n_blocks = num_blocks
        self.lambda_uncertainty = lambda_uncertainty
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.gridsize = gridsize

        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)
        time_embed_dim = self.hidden_size * 4
        self.time_embed = nn.Sequential(nn.Linear(self.hidden_size, time_embed_dim), SiLU(),
                                        nn.Linear(time_embed_dim, self.hidden_size))
        self.fuse_linear = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.att = Bi_Mamba(
            emb_dim=self.hidden_size,
            layer_num=self.n_blocks,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            drop_prob=self.dropout,
            gridsize=self.gridsize
        )
        self.drop_layer = nn.Dropout(self.dropout)
        self.norm_diffu_rep = LayerNorm(self.hidden_size)

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timesteps.device)
        config = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(config), torch.sin(config)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, rep_item, x_t, t, mask_seq):
        emb_t = self.time_embed(self.timestep_embedding(t, self.hidden_size))
        x_t = x_t + emb_t
        lambda_uncertainty = torch.normal(mean=torch.full(rep_item.shape, self.lambda_uncertainty),
                                          std=torch.full(rep_item.shape, self.lambda_uncertainty)).to(x_t.device)
        rep_diffu = self.att(rep_item + lambda_uncertainty * x_t.unsqueeze(1))
        rep_diffu = self.norm_diffu_rep(self.drop_layer(rep_diffu))
        out = rep_diffu[:, -1, :]
        return out, rep_diffu


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.
        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([self.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [self.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([self.num_timesteps], dtype=int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class FixSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.concatenate([np.ones([num_timesteps // 2]), np.zeros([num_timesteps // 2]) + 0.5])

    def weights(self):
        return self._weights


def create_named_schedule_sampler(name, num_timesteps):
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == "lossaware":
        return LossSecondMomentResampler(num_timesteps)
    elif name == "fixstep":
        return FixSampler(num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class DiffFormer(nn.Module):
    def __init__(self, hidden_size, schedule_sampler_name, diffusion_steps, noise_schedule, rescale_timesteps, device,
                 dropout, num_blocks, lambda_uncertainty, d_state, d_conv, expand, gridsize):
        super(DiffFormer, self).__init__()
        self.hidden_size = hidden_size
        self.schedule_sampler_name = schedule_sampler_name
        self.diffusion_steps = diffusion_steps
        self.noise_schedule = noise_schedule
        self.rescale_timesteps = rescale_timesteps
        self.device = device
        self.dropout = dropout
        self.n_blocks = num_blocks
        self.lambda_uncertainty = lambda_uncertainty
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.gridsize = gridsize
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = (betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = (betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.num_timesteps = int(self.betas.shape[0])
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)
        self.timestep_map = self.time_map()
        self.original_num_steps = len(betas)
        self.xstart_model = Diffu_xstart(
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_blocks=self.n_blocks,
            lambda_uncertainty=self.lambda_uncertainty,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            gridsize=self.gridsize
        )

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        return betas

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        x_t = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

        if mask == None:
            return x_t
        else:
            mask = torch.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return torch.where(mask == 0, x_start, x_t)

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):

        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output, _ = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        x_0 = model_output
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        device = next(self.xstart_model.parameters()).device
        indices = list(range(self.num_timesteps))[::-1]

        for i in indices:
            t = torch.tensor([i] * item_rep.shape[0], device=device)
            with torch.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t

    def forward(self, item_rep, item_tag, mask_seq):
        noise = torch.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], self.device)
        x_t = self.q_sample(item_tag, t, noise=noise)
        x_0, item_rep_out = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq)
        return x_0, item_rep_out, weights, t


class Magda(SequentialRecommender):
    def __init__(self, config, dataset):
        super(Magda, self).__init__(config, dataset)
        self.emb_dim = config['hidden_size']
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(config['emb_dropout'])
        self.LayerNorm = LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = config['dropout']

        self.n_blocks = config['num_blocks']
        self.lambda_uncertainty = config['lambda_uncertainty']
        self.d_state = config['d_state']
        self.d_conv = config['d_conv']
        self.expand = config['expand']
        self.gridsize = config['gridsize']
        self.loss_weight = config['loss_weight']
        self.preference_weight = config['preference_weight']

        self.hidden_size = config['hidden_size']
        self.schedule_sampler_name = config['schedule_sampler_name']
        self.diffusion_steps = config['diffusion_steps']
        self.noise_schedule = config['noise_schedule']
        self.rescale_timesteps = config['rescale_timesteps']
        self.diffformer = DiffFormer(
            self.hidden_size,
            self.schedule_sampler_name,
            self.diffusion_steps,
            self.noise_schedule,
            self.rescale_timesteps,
            self.device,
            self.dropout,
            self.n_blocks,
            self.lambda_uncertainty,
            self.d_state,
            self.d_conv,
            self.expand,
            self.gridsize
        )
        self.Bi_Mamba = Bi_Mamba(
            emb_dim=self.hidden_size,
            layer_num=self.n_blocks,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            drop_prob=self.dropout,
            gridsize=self.gridsize
        )
        self.loss_ce = nn.CrossEntropyLoss()
        self.ano_loss_ce = nn.CrossEntropyLoss()
        self.time_soft_attention = seq_affinity_soft_attention(emb_dim=self.hidden_size)

    def diffformer_pre(self, item_rep, tag_emb, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t = self.diffformer(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t

    def reverse(self, item_rep, noise_x_t, mask_seq):
        reverse_pre = self.diffformer.reverse_p_sample(item_rep, noise_x_t, mask_seq)
        return reverse_pre

    def loss_diffu_ce(self, rep_diffu, labels):
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def routing_rep_pre(self, rep_diffu):
        item_norm = (self.item_embeddings.weight ** 2).sum(-1).view(-1, 1)
        rep_norm = (rep_diffu ** 2).sum(-1).view(-1, 1)
        sim = torch.matmul(rep_diffu, self.item_embeddings.weight.t())
        dist = rep_norm + item_norm.transpose(0, 1) - 2.0 * sim
        dist = torch.clamp(dist, 0.0, np.inf)

        return -dist

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep / seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep / item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep / seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1) / torch.sum(mask_seq, dim=-1))

    def forward(self, item_seq, item_seq_len):
        item_embeddings = self.item_embeddings(item_seq)
        item_embeddings = self.embed_dropout(item_embeddings)
        item_embeddings = self.LayerNorm(item_embeddings)
        item_embeddings = self.Bi_Mamba(item_embeddings)
        time_seq_mask = item_seq.gt(0).unsqueeze(2).expand_as(item_embeddings)
        time_seq_mean = torch.mean(time_seq_mask * item_embeddings, dim=1)
        time_short = self.gather_indexes(item_embeddings, item_seq_len - 1)
        time_session = self.time_soft_attention(
            mask=time_seq_mask,
            short_feature=time_short,
            seq_feature=item_embeddings,
            affinity_feature=time_seq_mean
        )
        mask_seq = (item_seq > 0).float()
        return time_session, item_embeddings, mask_seq

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_session, item_embeddings, mask_seq = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        pos_items_emb = self.item_embeddings(pos_items.squeeze(-1))
        rep_diffu, _, _, _ = self.diffformer_pre(item_embeddings, pos_items_emb, mask_seq)
        scores = torch.matmul(rep_diffu, self.item_embeddings.weight.transpose(0, 1))
        loss = self.loss_ce(scores, pos_items)
        ano_scores = torch.matmul(time_session, self.item_embeddings.weight.transpose(0, 1))
        ano_loss = self.ano_loss_ce(ano_scores, pos_items)
        return self.loss_weight * loss + (1 - self.loss_weight) * ano_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_session, item_embeddings, mask_seq = self.forward(item_seq, item_seq_len)
        noise_x_t = torch.randn_like(item_embeddings[:, -1, :])
        rep_diffu = self.reverse(item_embeddings, noise_x_t, mask_seq)
        scores = torch.matmul((self.preference_weight * rep_diffu + (1 - self.preference_weight) * time_session),
                              self.item_embeddings.weight.transpose(0, 1))
        return scores
