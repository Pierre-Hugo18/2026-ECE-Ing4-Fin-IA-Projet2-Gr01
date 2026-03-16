import os
import json
import numpy as np
import gymnasium as gym
 
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
 
# ── Dossiers de sortie ────────────────────────────────────────────────────────
os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)
 
ENV_ID      = "CartPole-v1"
TOTAL_STEPS = 50_000
SEED        = 42
 
 
# ── Callback : enregistre le reward de chaque épisode ────────────────────────
class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
 
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True
 
 
# ── Fonction d'entraînement générique ─────────────────────────────────────────
def train(algo_name, model_class, hyperparams):
    print(f"\n{'='*50}")
    print(f"  Entraînement : {algo_name} sur {ENV_ID}")
    print(f"{'='*50}")
 
    env      = Monitor(gym.make(ENV_ID))
    callback = RewardLogger()
 
    model = model_class(
        "MlpPolicy",
        env,
        seed=SEED,
        verbose=0,
        **hyperparams,
    )
 
    model.learn(total_timesteps=TOTAL_STEPS, callback=callback)
    print(f"  Entraînement terminé ({TOTAL_STEPS} steps)")
 
    # Sauvegarde modèle
    model_path = os.path.join("models", algo_name)
    model.save(model_path)
    print(f"  Modèle sauvegardé  : {model_path}.zip")
 
    # Sauvegarde rewards
    rewards_path = os.path.join("results", f"{algo_name}_rewards.json")
    with open(rewards_path, "w") as f:
        json.dump(callback.episode_rewards, f)
    print(f"  Rewards sauvegardés : {rewards_path}")
 
    env.close()
    return callback.episode_rewards
 
 
# ── Hyperparamètres ───────────────────────────────────────────────────────────
DQN_PARAMS = {
    "learning_rate"          : 1e-3,
    "buffer_size"            : 50_000,
    "learning_starts"        : 1_000,
    "batch_size"             : 64,
    "gamma"                  : 0.99,
    "train_freq"             : 4,
    "target_update_interval" : 250,
    "exploration_fraction"   : 0.2,
    "exploration_final_eps"  : 0.05,
}
 
PPO_PARAMS = {
    "learning_rate" : 3e-4,
    "n_steps"       : 2048,
    "batch_size"    : 64,
    "n_epochs"      : 10,
    "gamma"         : 0.99,
    "gae_lambda"    : 0.95,
    "clip_range"    : 0.2,
    "ent_coef"      : 0.0,
}
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dqn_rewards = train("DQN", DQN, DQN_PARAMS)
    ppo_rewards = train("PPO", PPO, PPO_PARAMS)
 
    print("\n✅ Entraînement terminé !")
    print(f"   DQN — reward moyen (100 derniers épisodes) : {np.mean(dqn_rewards[-100:]):.1f}")
    print(f"   PPO — reward moyen (100 derniers épisodes) : {np.mean(ppo_rewards[-100:]):.1f}")
    print("\n→ Lance maintenant : python plot_results.py")