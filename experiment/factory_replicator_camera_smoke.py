#!/usr/bin/env python3
"""Smoke-test RGB capture in the Direct Factory peg-insert task.

The task deliberately does not use IsaacLab CameraCfg: it attaches Replicator
render products only after the stable DirectRLEnv has been created.
"""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import gymnasium as gym
    import numpy as np
    import omni.replicator.core as rep
    import torch
    from PIL import Image

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
    env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
    unwrapped = env.unwrapped
    env.reset()

    cameras = {
        "table": ((1.10, 0.00, 0.80), (0.00, 0.00, 0.30)),
        "side": ((0.65, -0.85, 0.52), (0.00, 0.00, 0.30)),
    }
    annotators = {}
    for name, (translation, target) in cameras.items():
        camera = rep.create.camera(position=translation, look_at=target)
        product = rep.create.render_product(camera, resolution=(84, 84))
        if not isinstance(product, str):
            product = product.path
        annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        annotator.attach(product)
        annotators[name] = annotator
        print(f"[FACTORY_REP] camera={name} product={product}", flush=True)

    action = torch.zeros((1, 6), dtype=torch.float32, device=unwrapped.device)
    for step in range(12):
        env.step(action)
        unwrapped.sim.render()
        descriptions = []
        for name, annotator in annotators.items():
            image = np.asarray(annotator.get_data())
            if step == 2:
                Image.fromarray(image[..., :3].astype(np.uint8)).save(f"/tmp/factory_rep_{name}.png")
            descriptions.append(f"{name}:shape={image.shape},size={image.size},mean={(float(image.mean()) if image.size else -1):.2f}")
        print(f"[FACTORY_REP] step={step} " + " | ".join(descriptions), flush=True)
    env.close()
    simulation_app.close()
except BaseException as exc:
    print(f"[FACTORY_REP] exception={type(exc).__name__}: {exc!r}", flush=True)
    traceback.print_exc()
    try:
        simulation_app.close()
    except Exception:
        pass
    raise
