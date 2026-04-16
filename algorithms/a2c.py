import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class A2CAgent:
    def __init__(
        self,
        policy_net,
        value_net,
        policy_optimizer,
        value_optimizer,
        gamma=0.99,
        entropy_coef=0.01,
        grad_clip=1.0,
        device="cpu"
    ):
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self.policy_net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value_net(state).squeeze(1)

        return action.item(), log_prob.squeeze(), entropy.squeeze(), value.squeeze()

    def compute_returns(self, rewards):
        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns

    def update(self, log_probs, entropies, values, rewards):
        returns = self.compute_returns(rewards)

        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values)

        if len(returns) > 1:
            returns_for_value = (returns - returns.mean()) / (returns.std() + 1e-8)
        else:
            returns_for_value = returns

        advantages = returns - values.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()
        total_actor_loss = actor_loss + entropy_loss

        value_loss = F.smooth_l1_loss(values, returns_for_value)

        self.policy_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
        self.value_optimizer.step()

        return (
            total_actor_loss.item(),
            value_loss.item(),
            advantages.mean().item()
        )