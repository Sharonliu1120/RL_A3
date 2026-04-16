import torch
from torch.distributions import Categorical


class ReinforceAgent:
    def __init__(self, policy_net, optimizer, gamma=0.99, device="cpu"):
        self.policy_net = policy_net
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def compute_returns(self, rewards):
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return returns

    def update(self, log_probs, rewards):
        returns = self.compute_returns(rewards)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss += -log_prob * G

        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()