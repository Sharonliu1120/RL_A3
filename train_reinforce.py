import os
import csv
import torch
import torch.optim as optim

from models.policy_network import PolicyNetwork
from algorithms.reinforce import ReinforceAgent
from utils.env import make_env
from utils.seed import set_seed
from utils.plot import plot_reinforce_interpolated


def train_single_seed(
    env_name="CartPole-v1",
    episodes=1500,
    gamma=0.99,
    lr=5e-4,
    hidden_dim=128,
    seed=42,
    device="cpu"
):
    set_seed(seed)
    env = make_env(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    agent = ReinforceAgent(policy_net, optimizer, gamma, device)

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/reinforce_seed_{seed}.csv"

    total_steps = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "env_steps", "return", "loss"])

        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            done = False

            log_probs = []
            rewards = []
            episode_return = 0

            while not done:
                action, log_prob = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)

                state = next_state
                episode_return += reward
                total_steps += 1

            loss = agent.update(log_probs, rewards)

            writer.writerow([episode, total_steps, episode_return, loss])

            if episode % 50 == 0:
                print(
                    f"Seed {seed} | Episode {episode} | "
                    f"Return: {episode_return} | Loss: {loss:.4f}"
                )

    env.close()
    torch.save(policy_net.state_dict(), f"results/reinforce_model_seed_{seed}.pt")
    print(f"Finished seed {seed}. Results saved to {csv_path}")


def train_multiple_seeds():
    seeds = [42, 123, 999]

    for seed in seeds:
        train_single_seed(seed=seed)

    plot_reinforce_interpolated(
        csv_paths=[f"results/reinforce_seed_{seed}.csv" for seed in seeds],
        output_path="results/reinforce_interp.png"
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_multiple_seeds()