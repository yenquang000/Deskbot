"""Run N practice episodes per policy on the same seeds and report the numbers.

  python evaluate.py --episodes 200 --model deskbot_final.zip
Writes results.json (read by make_chart.py).
"""
import argparse
import json

import numpy as np

from baseline import EdgeShield, LawnmowerPolicy
from env import DeskbotEnv


class RandomPolicy:
    def __init__(self, seed=0):
        self.g = np.random.default_rng(seed)

    def reset(self):
        pass

    def predict(self, obs):
        return int(self.g.integers(4))


class PPOPolicy:
    def __init__(self, path):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def reset(self):
        pass

    def predict(self, obs):
        return int(self.model.predict(obs, deterministic=True)[0])


def run(policy, episodes, stage, shield=False, seed0=10_000):
    env = DeskbotEnv(stage=stage)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        policy.reset()
        sh = EdgeShield() if shield else None
        done, ret, info = False, 0.0, {}
        while not done:
            a = policy.predict(obs)
            if sh:
                a = sh(obs, a)
            obs, r, term, trunc, info = env.step(a)
            ret += r
            done = term or trunc
        rows.append(dict(ret=ret, steps=env.t, coverage=info["coverage"],
                         edge_hit=int(info["edge_hit"]), success=int(info.get("success", False)),
                         debris=1 - info["debris_left"] / max(env.n_debris, 1) if env.n_debris else 1.0,
                         pens=1 - info["pens_left"] / max(env.n_pens, 1) if env.n_pens else 1.0))
    keys = rows[0].keys()
    out = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    out["edge_hit_episodes"] = int(sum(r["edge_hit"] for r in rows))
    out["episodes"] = episodes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    results = {
        "random": run(RandomPolicy(), a.episodes, a.stage),
        "backup pattern": run(LawnmowerPolicy(), a.episodes, a.stage),
        "backup + edge guard": run(LawnmowerPolicy(), a.episodes, a.stage, shield=True),
    }
    if a.model:
        pol = PPOPolicy(a.model)
        results["PPO model"] = run(pol, a.episodes, a.stage)
        results["PPO + edge guard"] = run(pol, a.episodes, a.stage, shield=True)

    print(f"{'policy':22s} {'success':>8s} {'cover':>7s} {'debris':>7s} {'pens':>6s} {'steps':>7s} {'edge-hit eps':>13s}")
    for k, v in results.items():
        print(f"{k:22s} {v['success']:8.2f} {v['coverage']:7.2f} {v['debris']:7.2f} {v['pens']:6.2f} "
              f"{v['steps']:7.0f} {v['edge_hit_episodes']:>8d}/{v['episodes']}")
    if a.model:
        b, m = results["backup + edge guard"], results["PPO + edge guard"]
        ok = (m["edge_hit_episodes"] == 0 and m["success"] >= b["success"]
              and (m["success"] > b["success"] or m["steps"] <= b["steps"]))
        print("\nDay-8 criterion (beats backup, zero edge hits in", a.episodes, "episodes):",
              "PASS" if ok else "NOT YET")
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
