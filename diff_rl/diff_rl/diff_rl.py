from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import functools
import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic

from diff_rl.common.buffers import ReplayBuffer
from diff_rl.common.off_policy_algorithm import OffPolicyAlgorithm
from diff_rl.diff_rl.policies import Actor, MlpPolicy, TD3Policy

SelfTD3 = TypeVar("SelfTD3", bound="TD3")


class TD3(OffPolicyAlgorithm):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self):
        super()._setup_model() 
        self._create_aliases()

        self.target_entropy = float(-np.sqrt(np.prod(self.env.action_space.shape)).astype(np.float32))

        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

        init_value = 1.0
        self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
        self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))

    def _create_aliases(self):
        # TODO, check actor_target here
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 100):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to lr schedule
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            sampled_action = self.consistency_model.sample(model=self.actor, state=replay_data.observations)
            log_prob = self.consistency_model.kde_prob(state=replay_data.next_observations, 
                                                        action=sampled_action, 
                                                        model=self.actor)
                                                        
            ent_coef_loss = None
            ent_coef = th.exp(self.log_ent_coef.detach()) # TODO, adjust the target_entropy here
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            ent_coef_losses.append(ent_coef_loss.item())

            ent_coefs.append(ent_coef.item())

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy and add clipped noise
                # noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                # noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip) # here is the [-1, 1] action

                # next_actions = (self.consistency_model.sample(model=self.actor_target, state=replay_data.next_observations) + noise).clamp(-1, 1)
                next_actions = self.consistency_model.sample(model=self.actor_target, state=replay_data.next_observations)
                next_log_prob = self.consistency_model.kde_prob(state=replay_data.next_observations, 
                                                               action=next_actions, 
                                                               model=self.actor)
                
                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                # next_q_values = next_q_values - ent_coef * next_log_prob
                next_q_values = next_q_values
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values) # sum two loss from two critic, update them all
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic.optimizer.step()

            # x_start_noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip) # here is the [-1, 1] action
            # x_start = (self.consistency_model.sample(model=self.actor_target, state=replay_data.observations) + x_start_noise).clamp(-1, 1)
            sampled_action = self.consistency_model.sample(model=self.actor, state=replay_data.observations)
            compute_bc_losses = functools.partial(self.consistency_model.consistency_losses,
                                            model=self.actor,
                                            x_start=replay_data.actions,
                                            num_scales=40,
                                            target_model=self.actor_target,
                                            state=replay_data.observations,
                                            )
            bc_losses = compute_bc_losses() # but here take loss rather than consistency_loss

            q_values_pi = th.cat(self.critic(replay_data.observations, sampled_action), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            # actor_loss = (bc_losses["consistency_loss"] - min_qf_pi - ent_coef * log_prob).mean()
            actor_loss = (bc_losses["consistency_loss"] - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())
                
            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int = None,
        callback: MaybeCallback = None,
        log_interval: int = 10,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=self.buffer_size,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self):
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
    
    def test(self, env):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        # 这里假设 action 是 50x3 的张量，代表 50 个 3D 坐标
        state = env.reset()
        with th.no_grad():
            state = th.FloatTensor(state.reshape(1, -1)).to(self.device)
            state_rpt = th.repeat_interleave(state, repeats=10000, dim=0)
            action = self.consistency_model.sample(model=self.actor, state=state_rpt)
            q_value = self.critic.q1_forward(state_rpt, action).flatten()

        # 将 action 转换为 numpy 数组（如果是 tensor）
        action = action.cpu().numpy()
        a = 1

        # 分解动作的 x, y, z 坐标
        x = action[:, 0]
        y = action[:, 3]
        z = action[:, 7]

        # 创建 3D 图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 根据每个动作的模长来设置颜色（你也可以根据其他属性来设定）
        colors = np.linalg.norm(action, axis=1)  # 计算每个动作的模长
        sc = ax.scatter(x, y, z, c=colors, cmap='viridis', marker='o')

        # 设置 x, y, z 轴的范围 [-1, 1]
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # 添加颜色条
        plt.colorbar(sc)

        # 添加标题
        ax.set_title('3D Action Visualization')
        print(q_value)
        # 显示图形
        plt.show()
        a = 1
