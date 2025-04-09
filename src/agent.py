import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from torchsummary import summary

# ---
# Function for initializing Linear layer and creating following original's PPO implementation
# ---
def __init_linear_layer_ppo(layer: nn.Linear, std: float, bias_const: float) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std) # type: ignore
    torch.nn.init.constant_(layer.bias, bias_const) # type: ignore
    return layer

def make_linear_layer(in_dim: int, out_dim: int, std: float = np.sqrt(2), bias_const: float = 0) -> nn.Linear:
    layer = nn.Linear(in_features=in_dim, out_features=out_dim, dtype=torch.float32)
    layer = __init_linear_layer_ppo(layer, std, bias_const)
    return layer


# -------------
# Cheetah Agent
# -------------
class CheetahAgent(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        device: torch.device, 
        verbose=False,
    ):
        super(CheetahAgent, self).__init__()

        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_dist_mean = nn.Sequential(
            make_linear_layer(in_dim=state_dim, out_dim=64),
            nn.Tanh(),
            make_linear_layer(in_dim=64, out_dim=64),
            nn.Tanh(),
            make_linear_layer(in_dim=64, out_dim=action_dim, std=1e-2),
            nn.Tanh(),
        )
        self.actor_dist_logstd = nn.Parameter(torch.zeros(size=(action_dim,), dtype=torch.float32))

        self.critic = nn.Sequential(
            make_linear_layer(in_dim=state_dim, out_dim=64),
            nn.Tanh(),
            make_linear_layer(in_dim=64, out_dim=64),
            nn.Tanh(),
            make_linear_layer(in_dim=64, out_dim=1, std=1),
        )

        if verbose:
            print('================================================================')
            print(f'==                       POLICY NETWORK                       ==')
            print('================================================================')
            print(f'--         NN to predict mean of action distributions         --')
            summary(self.actor_dist_mean, (state_dim,))
            print(f'--  Trainable params to predict std of action distributions   --')
            print(f'Parameters shape: \t {self.actor_dist_logstd.shape}')
            print(f'Number of trainable parameters: \t {self.actor_dist_logstd.numel()}')
            print('\n================================================================')
            print(f'==                       VALUE NETWORK                        ==')
            print('================================================================')
            summary(self.critic, (state_dim,))
            

        self.actor_dist_mean = self.actor_dist_mean.to(self.device)
        self.actor_dist_logstd.data = self.actor_dist_logstd.data.to(self.device)
        self.critic = self.critic.to(self.device)

    def _get_action_distribution(self, state: torch.Tensor) -> dist.Normal:
        action_dist_mean = self.actor_dist_mean(state)
        action_dist_std = torch.exp(self.actor_dist_logstd.expand_as(action_dist_mean))
        action_dist = dist.Normal(action_dist_mean, action_dist_std)
        return action_dist

    def get_action_distribution_entropy(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        action_dist = self._get_action_distribution(state)
        action_dist_entropy = action_dist.entropy().sum(dim=-1)
        return action_dist_entropy

    def get_logprobability_for_chosen_action(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
    ) -> torch.Tensor:
        state = state.to(self.device)
        action_dist = self._get_action_distribution(state)
        action_logprobs = action_dist.log_prob(action).sum(dim=-1)
        return action_logprobs
    
    def choose_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = state.to(self.device)
        action_dist = self._get_action_distribution(state)
        action = action_dist.sample()
        action_logprobs = action_dist.log_prob(action).sum(dim=-1)
        return action, action_logprobs
    
    def evaluate_state_value(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        value = self.critic(state).squeeze(dim=-1)
        return value