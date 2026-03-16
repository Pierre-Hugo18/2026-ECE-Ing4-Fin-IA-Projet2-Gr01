import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
 
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "comparaison_DQN_PPO.png")
 
 
# ── Utilitaires ───────────────────────────────────────────────────────────────
def load_rewards(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [ERREUR] Fichier introuvable : {path}")
        print("  → Lance d'abord train.py et random_agent.py")
        return []
    with open(path) as f:
        return json.load(f)
 
 
def smooth(rewards, window=20):
    """Moyenne glissante pour lisser les courbes."""
    if len(rewards) < window:
        return rewards
    return (
        np.convolve(rewards, np.ones(window) / window, mode="valid").tolist()
    )
 
 
def summary(name, rewards):
    if not rewards:
        return
    arr = np.array(rewards)
    print(f"  {name:10s} | épisodes : {len(arr):5d} | "
          f"moy : {arr.mean():7.1f} | "
          f"max : {arr.max():7.1f} | "
          f"derniers 50 : {arr[-50:].mean():7.1f}")
 
 
# ── Chargement des données ────────────────────────────────────────────────────
print("── Chargement des rewards ──────────────────────────────────")
dqn_raw    = load_rewards("DQN_rewards.json")
ppo_raw    = load_rewards("PPO_rewards.json")
random_raw = load_rewards("random_rewards.json")
 
print("\n── Résumé statistique ──────────────────────────────────────")
summary("DQN",    dqn_raw)
summary("PPO",    ppo_raw)
summary("Random", random_raw)
 
 
# ── Tracé ─────────────────────────────────────────────────────────────────────
WINDOW = 20
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CartPole-v1 — Comparaison DQN vs PPO vs Random",
             fontsize=14, fontweight="bold", y=1.01)
 
COLORS = {"DQN": "#2196F3", "PPO": "#4CAF50", "Random": "#9E9E9E"}
 
datasets = [
    ("DQN",    dqn_raw),
    ("PPO",    ppo_raw),
    ("Random", random_raw),
]
 
# ── Graphe 1 : reward brut par épisode ───────────────────────────────────────
ax1 = axes[0]
for name, raw in datasets:
    if not raw:
        continue
    x = np.arange(len(raw))
    ax1.plot(x, raw, alpha=0.15, color=COLORS[name], linewidth=0.8)
    smoothed = smooth(raw, WINDOW)
    x_s = np.arange(len(smoothed))
    ax1.plot(x_s, smoothed, label=f"{name} (lissé ×{WINDOW})",
             color=COLORS[name], linewidth=2)
 
ax1.set_title("Reward par épisode")
ax1.set_xlabel("Épisode")
ax1.set_ylabel("Reward total")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
 
# Ligne objectif CartPole (score parfait = 500)
ax1.axhline(500, color="gold", linewidth=1, linestyle="--", alpha=0.6, label="Score max (500)")
 
# ── Graphe 2 : moyenne glissante sur 50 épisodes ─────────────────────────────
ax2 = axes[1]
WINDOW2 = 50
for name, raw in datasets:
    if not raw:
        continue
    smoothed = smooth(raw, WINDOW2)
    x_s = np.arange(len(smoothed))
    ax2.plot(x_s, smoothed, label=name, color=COLORS[name], linewidth=2.5)
 
ax2.set_title(f"Moyenne glissante ({WINDOW2} épisodes)")
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Reward moyen")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(500, color="gold", linewidth=1, linestyle="--", alpha=0.6)
 
plt.tight_layout()
 
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"\n✅ Figure sauvegardée : {OUTPUT_FILE}")
plt.show()