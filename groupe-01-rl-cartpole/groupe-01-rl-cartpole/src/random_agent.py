import os
import json
import numpy as np
import gymnasium as gym
 
# ── Config ────────────────────────────────────────────────────────────────────
ENV_ID     = "CartPole-v1"
N_EPISODES = 200
SEED       = 42
 
os.makedirs("results", exist_ok=True)
 
 
# ── Évaluation de la politique aléatoire ─────────────────────────────────────
def run_random_agent(env_id, n_episodes, seed):
    env = gym.make(env_id)
    rewards = []
 
    print(f"Agent aléatoire sur {env_id} — {n_episodes} épisodes...")
 
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
 
        while not done:
            action = env.action_space.sample()          # action totalement aléatoire
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
 
        rewards.append(total_reward)
 
        if (ep + 1) % 50 == 0:
            print(f"  Épisode {ep+1:3d}/{n_episodes} — "
                  f"reward moy. jusqu'ici : {np.mean(rewards):.1f}")
 
    env.close()
    return rewards
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rewards = run_random_agent(ENV_ID, N_EPISODES, SEED)
 
    # Sauvegarde
    output_path = os.path.join("results", "random_rewards.json")
    with open(output_path, "w") as f:
        json.dump(rewards, f)
 
    # Résumé statistique
    arr = np.array(rewards)
    print(f"\n── Résumé agent aléatoire ──────────────────────────────")
    print(f"  Épisodes      : {len(arr)}")
    print(f"  Reward moyen  : {arr.mean():.2f}")
    print(f"  Reward max    : {arr.max():.2f}")
    print(f"  Reward min    : {arr.min():.2f}")
    print(f"  Écart-type    : {arr.std():.2f}")
    print(f"\n✅ Rewards sauvegardés : {output_path}")
    print("\n→ Lance maintenant : python train.py")