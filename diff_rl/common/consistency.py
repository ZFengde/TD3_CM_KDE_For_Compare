
import torch as th
import numpy as np
import torch.nn.functional as F

from diff_rl.common.helpers import append_dims, append_zero, mean_flat
from stable_baselines3.common.preprocessing import get_action_dim

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class Consistency_Model:
    def __init__(
        self,
        env,
        device,
        sigma_data: float = 1,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        ts=None,
        sampler="onestep", 
    ):
        self.env = env
        self.action_dim = get_action_dim(self.env.action_space)
        self.action_low, self.action_high = self.env.action_space.low[0], self.env.action_space.high[0]

        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.num_timesteps = 40

        self.device = device
        
        self.sampler = sampler
        self.steps = steps
        self.ts = [0, 20, 40]
        # this is for sample
        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings_for_boundary_condition(self, sigma): # get the values of c_...
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)

    def consistency_losses(
        self,
        model,
        x_start, # batch * action_dim
        num_scales,
        state=None, # # batch * obs_dim
        target_model=None,
        noise=None,
    ):
        noise = th.randn_like(x_start) # make this noise a \nabla[Q(s, a)]
        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(model, x, t, state)[1]

        @th.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        ) # random 

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        ) # t (t_n+1) and t2(t_n) here is randomly generated based on indices
        t = t**self.rho # t_n+1 

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho # t_n
        
        x_t = x_start + noise * append_dims(t, dims) # basically is x_start + noise

        dropout_state = th.get_rng_state() # random number generator
        distiller = denoise_fn(x_t, t, state) # # predicted target based on t = t_n+1

        x_t2 = x_start + noise * append_dims(t2, dims).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state) # predicted target based on t2=t_n
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t) # sigmas**-2
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data) # lambda(t_n), get different weights based on snrs: snrs + 1.0 / sigma_data**-2
        # t 越小， weights越大
        consistency_diffs = (distiller - distiller_target) ** 2 # get the consistency difference
        consistency_loss = mean_flat(consistency_diffs) * weights # weighted average as loss

        # === 新增对比损失，基于不同噪声 === 
        # Contrastive loss, based on different state-action pair
        # TODO, use F.cosine_similarity(z_i, z_j, dim=-1) here
        # noise_diff = th.randn_like(x_start)  # 生成不同的噪声
        noise_diff = - noise # 生成不同的噪声
        x_t_diff = x_start + noise_diff * append_dims(t, dims).detach()
        
        distiller_diff = target_denoise_fn(x_t_diff, t, state) # 计算不同噪声下生成的动作
        distiller_diff = distiller_diff.detach()
        noise_similarity = F.cosine_similarity(noise, noise_diff, dim=-1)  
        output_distance = (distiller - distiller_diff) ** 2 # 对比损失，噪声差异越大，输出的差异也应该越大
        # [-1, 1], 1: same, -1: reverse, 0: vertical, should think about the coefficient here
        # TODO, we should consider the timestep, noise_similarity as well as the norm here
        # contrastive_loss = mean_flat(output_distance) * weights * noise_similarity # weighted average as loss
        contrastive_weights = F.softmax(-weights, dim=0)
        contrastive_loss = - mean_flat(output_distance) * contrastive_weights # weighted average as loss

        terms = {}
        terms["consistency_loss"] = consistency_loss
        terms["contrastive_loss"] = contrastive_loss

        return terms

    def denoise(self, model, x_t, sigmas, state):
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        
        model_output = model(c_in * x_t, rescaled_t, state)
        denoised = c_out * model_output + c_skip * x_t 
        denoised = denoised.clamp(-1, 1)
        # since the maximum of c_out is 0.5, output from model is 1
        # then denoised in [-0.5, 0.5]
        return model_output, denoised

    def sample(self, model, state): # 
        x_0 = self.sample_onestep(model, state)
        return x_0
    
    def sample_onestep(self, model, state):
        x_T = th.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]]) # stands for sigma input
        return self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
    
    def sample_multistep(self, model, state):
        x_T = th.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max

        t_max_rho = self.sigma_max ** (1 / self.rho)
        t_min_rho = self.sigma_min ** (1 / self.rho)
        s_in = x_T.new_ones([x_T.shape[0]])

        x = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]

        for i in range(len(self.ts)):
            t = (t_max_rho + self.ts[i] / (self.steps - 1) * (t_min_rho - t_max_rho)) ** self.rho
            t = np.clip(t, self.sigma_min, self.sigma_max)
            x = self.denoise(model, x, t * s_in, state)[1]

        return x

    def batch_multi_sample(self, model, state): # 
        x_T = th.randn((state.shape[0], state.shape[1], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]]) # stands for sigma input
        x_0 = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
        return x_0
    
    def kde_log_prob(self, data, new_samples, bandwidth=0.5, epsilon=1e-8):
        """
        data: (batch_size, n_samples, n_features)
        new_samples: (batch_size, 1, n_features)
        bandwidth: Scalar value for the Gaussian kernel bandwidth
        epsilon: Small value to ensure numerical stability
        """
        # 计算新样本和数据的差值
        diff = new_samples - data  # 计算逐对差值
        distances = th.sum(diff**2, dim=-1)  # 欧氏距离的平方

        # 限制距离避免在计算 exp 时数值溢出
        distances = th.clamp(distances, max=1e8)
        kernel_values = th.exp(-distances / (2 * bandwidth**2))  # 高斯核函数

        # 计算归一化因子
        normalization_factor = (bandwidth * (2 * th.pi)**0.5)**data.shape[-1]

        # 计算概率，加入 epsilon 避免数值问题
        probs = kernel_values.sum(dim=-1) / (data.shape[1] * normalization_factor + epsilon)
        probs = th.clamp(probs, min=epsilon)  # 避免 log(0)

        log_probs = th.log(probs)  # 计算 log 概率
        return log_probs

    
    def kde_prob(self, state, action, model):
        state_rpt = th.repeat_interleave(state.unsqueeze(1), repeats=50, dim=1).detach()
        scaled_actions = self.batch_multi_sample(model=model, state=state_rpt).detach()
        log_probs = self.kde_log_prob(scaled_actions, action.unsqueeze(1), bandwidth=0.5)
        return log_probs.unsqueeze(1).detach()