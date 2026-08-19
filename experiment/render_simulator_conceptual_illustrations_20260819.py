#!/usr/bin/env python3
"""Render four scripted Isaac Sim Factory peg-in-hole conceptual illustrations.

These frames are deliberately not policy rollouts or Formal64 evaluation data.
The scene is initialized once, then the robot/held peg are placed at fixed
illustrative waypoints while the fixture, lighting, and camera remain fixed.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--snapshot", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import carb
carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import omni.usd
import torch
from PIL import Image
from pxr import Gf, UsdLux, UsdGeom
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def settle(e, steps=14):
    for _ in range(steps):
        app.update()
        e.sim.render()


def restore(e, s):
    e._robot.write_root_pose_to_sim(s["robot_root"][:, :7])
    e._robot.write_root_velocity_to_sim(s["robot_root"][:, 7:])
    e._robot.write_joint_state_to_sim(s["robot_joint_pos"], s["robot_joint_vel"])
    e._held_asset.write_root_pose_to_sim(s["held_root"][:, :7])
    e._held_asset.write_root_velocity_to_sim(s["held_root"][:, 7:])
    e._fixed_asset.write_root_pose_to_sim(s["fixed_root"][:, :7])
    e._fixed_asset.write_root_velocity_to_sim(s["fixed_root"][:, 7:])
    if "ctrl_target_joint_pos" in s:
        e._robot.set_joint_position_target(s["ctrl_target_joint_pos"])
    e.scene.write_data_to_sim()
    e.scene.update(dt=e.physics_dt)
    e.sim.render()
    settle(e)


cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = 20260819
cfg.episode_length_s = 30
cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
cfg.sim.render_interval = cfg.decimation
cfg.sim.use_fabric = True
env = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
e = env.unwrapped
env.reset(seed=20260819)
snapshot = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in torch.load(a.snapshot, map_location="cpu", weights_only=False).items()}
restore(e, snapshot)

# Use the standard Factory scene light plus low-intensity fill/key lights.  No
# labels or synthetic overlays are added to the rendered pixels.
stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/ConceptualDome")
dome.CreateIntensityAttr(1800.0)
dome.CreateColorAttr((0.82, 0.86, 0.92))
key = UsdLux.SphereLight.Define(stage, "/World/ConceptualKey")
key.CreateIntensityAttr(9000.0)
key.CreateRadiusAttr(0.12)
UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.92, -0.55, 1.18))
fill = UsdLux.DistantLight.Define(stage, "/World/ConceptualFill")
fill.CreateIntensityAttr(3500.0)
fill.CreateAngleAttr(0.35)
fill.CreateColorAttr((0.75, 0.82, 1.0))
carb.settings.get_settings().set_float("/rtx/post/tonemap/exposure", 0.5)

# Fixed three-quarter contact view selected from the rendered camera probe.
target = np.array([0.600, 0.000, 0.105], dtype=np.float32)
position = np.array([0.980, -0.750, 0.620], dtype=np.float32)
camera = rep.create.camera(position=tuple(position.tolist()), look_at=tuple(target.tolist()), focal_length=42.0, horizontal_aperture=20.955)
product = rep.create.render_product(camera, resolution=(1920, 1080))
ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
ann.attach(product if isinstance(product, str) else product.path)

# Translation waypoints are illustrative configurations only.  Both the robot
# root and held peg move together so the grasp stays visually coherent.
waypoints = {
    "01_initial_lateral_error": np.array([0.016, 0.009, 0.010], dtype=np.float32),
    "02_contact_misalignment": np.array([0.008, 0.004, 0.003], dtype=np.float32),
    "03_lateral_recovery": np.array([0.002, 0.001, 0.000], dtype=np.float32),
    "04_aligned_insertion": np.array([0.000, 0.000, -0.004], dtype=np.float32),
}
base_robot = e._robot.data.root_state_w[0].detach().clone()
base_held = e._held_asset.data.root_state_w[0].detach().clone()
out = a.output / "contact_recovery_sequence"
out.mkdir(parents=True, exist_ok=True)
for name, delta in waypoints.items():
    robot = base_robot.clone(); held = base_held.clone()
    robot[:3] += torch.tensor(delta, device="cuda")
    held[:3] += torch.tensor(delta, device="cuda")
    e._robot.write_root_pose_to_sim(robot[None, :7])
    e._held_asset.write_root_pose_to_sim(held[None, :7])
    e.scene.write_data_to_sim(); e.scene.update(dt=e.physics_dt); e.sim.render(); settle(e)
    pixels = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(out / f"{name}.png", format="PNG")

# Deterministic metadata required by the repository request.
meta = {
    "case_type": "simulator_rendered_conceptual_illustration",
    "formal64_pair_id": None,
    "is_formal64_evaluation_trajectory": False,
    "is_policy_output": False,
    "scene_seed": 20260819,
    "script_or_waypoint_config": "experiment/render_simulator_conceptual_illustrations_20260819.py",
    "source_scene": "Isaac Lab Factory PegInsert Direct scene with standard robot, factory peg, factory hole fixture, and table",
    "camera_pose": position.tolist(),
    "camera_target": target.tolist(),
    "resolution": [1920, 1080],
    "lighting": {"dome_intensity": 1800.0, "key_intensity": 9000.0, "fill_intensity": 3500.0, "tonemap_exposure": 0.5},
    "states": {name: {"step": i, "png": f"contact_recovery_sequence/{name}.png", "waypoint_delta_xyz_m": delta.tolist()} for i, (name, delta) in enumerate(waypoints.items())},
    "caption": "Simulator-rendered conceptual illustration of contact-stage recovery. The states are scripted illustrative configurations and are not Formal64 evaluation trajectories or quantitative outcomes.",
}
(a.output / "contact_recovery_sequence" / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
env.close(); app.close()
print(json.dumps({"output": str(a.output), "states": list(waypoints), "resolution": [1920, 1080]}))
