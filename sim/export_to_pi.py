"""Convert a trained SB3 PPO model into a small .npz the Pi can run with NumPy only
(no torch / stable-baselines3 needed on the robot).

  python export_to_pi.py models/final.zip deskbot_policy.npz
"""
import sys

import numpy as np


def export(model_path, out_path):
    import torch.nn as nn
    from stable_baselines3 import PPO

    model = PPO.load(model_path, device="cpu")
    pol = model.policy
    linears = [m for m in pol.mlp_extractor.policy_net if isinstance(m, nn.Linear)]
    if not isinstance(pol.mlp_extractor.policy_net[1], nn.Tanh):
        raise SystemExit("pi_inference.py assumes tanh activations; retrain with Tanh or edit it.")
    arrays = {}
    for i, lin in enumerate(linears):
        arrays[f"W{i}"] = lin.weight.detach().numpy().astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32)
    arrays["Wout"] = pol.action_net.weight.detach().numpy().astype(np.float32)
    arrays["bout"] = pol.action_net.bias.detach().numpy().astype(np.float32)
    arrays["n_hidden"] = np.array(len(linears))
    arrays["obs_dim"] = np.array(model.observation_space.shape[0])
    np.savez(out_path, **arrays)

    # verify against SB3 on random observations before shipping
    from pi_inference import DeskbotPolicy
    pi = DeskbotPolicy(out_path)
    g = np.random.default_rng(0)
    obs = g.normal(size=(200, pi.obs_dim)).astype(np.float32)
    ref = model.predict(obs, deterministic=True)[0]
    mine = np.array([pi.act(o) for o in obs])
    assert (ref == mine).all(), "export mismatch"
    print(f"Wrote {out_path}; matches SB3 on 200/200 random observations.")


if __name__ == "__main__":
    export(sys.argv[1], sys.argv[2])
