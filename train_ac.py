import os
import csv
import torch
import torch.optim as optim

from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from algorithms.ac import ActorCriticAgent
from utils.env import make_env
from utils.seed import set_seed
from utils.plot import plot_algorithm_interpolated


def train_single_seed(
    env_name="CartPole-v1",
    episodes=2000,
    gamma=0.99,
    policy_lr=3e-4,
    value_lr=1e-3,
    hidden_dim=128,
    entropy_coef=0.01,
    grad_clip=1.0,
    seed=42,
    device="cpu"
):
    set_seed(seed)
    env = make_env(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
    value_net = ValueNetwork(state_dim, hidden_dim).to(device)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    agent = ActorCriticAgent(
        policy_net=policy_net,
        value_net=value_net,
        policy_optimizer=policy_optimizer,
        value_optimizer=value_optimizer,
        gamma=gamma,
        entropy_coef=entropy_coef,
        grad_clip=grad_clip,
        device=device
    )

    os.makedirs("results", exist_ok=True)
    csv_path = f"results/ac_seed_{seed}.csv"

    total_steps = 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "env_steps",
            "return",
            "actor_loss",
            "critic_loss",
            "mean_advantage"
        ])

        for episode in range(1, episodes + 1):
            state, _ = env.reset(seed=seed + episode)
            done = False

            episode_return = 0
            rewards = []
            log_probs = []
            entropies = []
            values = []

            while not done:
                action, log_prob, entropy, value = agent.select_action(state)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)

                state = next_state
                episode_return += reward
                total_steps += 1

            actor_loss, critic_loss, mean_advantage = agent.update(
                log_probs=log_probs,
                entropies=entropies,
                values=values,
                rewards=rewards
            )

            writer.writerow([
                episode,
                total_steps,
                episode_return,
                actor_loss,
                critic_loss,
                mean_advantage
            ])

            if episode % 50 == 0:
                print(
                    f"Seed {seed} | Episode {episode} | "
                    f"Return: {episode_return:.1f} | "
                    f"Actor Loss: {actor_loss:.4f} | "
                    f"Critic Loss: {critic_loss:.4f} | "
                    f"Advantage: {mean_advantage:.4f}"
                )

    env.close()

    torch.save(policy_net.state_dict(), f"results/ac_policy_seed_{seed}.pt")
    torch.save(value_net.state_dict(), f"results/ac_value_seed_{seed}.pt")
    print(f"Finished seed {seed}. Results saved to {csv_path}")


def train_multiple_seeds():
    seeds = [42, 123, 999]

    for seed in seeds:
        train_single_seed(seed=seed)

    plot_algorithm_interpolated(
        csv_paths=[f"results/ac_seed_{seed}.csv" for seed in seeds],
        output_path="results/ac_interp.png",
        title="Actor-Critic (Interpolated, 3 seeds)"
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_multiple_seeds()