#!/usr/bin/env python3
"""Render matched contact-misalignment and lateral-recovery stills.

Both states use one camera and one lighting setup. The same USD translation is
applied to the complete Robot and HeldAsset roots because the Factory robot is
fixed-base. These are scripted conceptual illustrations, not policy outputs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--snapshot", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); app = AppLauncher(a).app

import carb
carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import omni.usd
import torch
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = 20260819; cfg.episode_length_s = 30
cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
cfg.sim.render_interval = cfg.decimation; cfg.sim.use_fabric = True
env_obj = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
env = env_obj.unwrapped; env_obj.reset(seed=20260819)
s = torch.load(a.snapshot, map_location="cpu", weights_only=False)
s = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in s.items()}
env._robot.write_root_pose_to_sim(s["robot_root"][:, :7])
env._robot.write_root_velocity_to_sim(s["robot_root"][:, 7:])
env._robot.write_joint_state_to_sim(s["robot_joint_pos"], s["robot_joint_vel"])
env._held_asset.write_root_pose_to_sim(s["held_root"][:, :7])
env._held_asset.write_root_velocity_to_sim(s["held_root"][:, 7:])
env._fixed_asset.write_root_pose_to_sim(s["fixed_root"][:, :7])
env._fixed_asset.write_root_velocity_to_sim(s["fixed_root"][:, 7:])
if "ctrl_target_joint_pos" in s: env._robot.set_joint_position_target(s["ctrl_target_joint_pos"])
env.scene.write_data_to_sim(); env.scene.update(dt=env.physics_dt); env.sim.render()

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/ConceptualDomePair")
dome.CreateIntensityAttr(1800.0); dome.CreateColorAttr((0.82, 0.86, 0.92))
key = UsdLux.SphereLight.Define(stage, "/World/ConceptualKeyPair")
key.CreateIntensityAttr(9000.0); key.CreateRadiusAttr(0.12)
UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.92, -0.55, 1.18))
fill = UsdLux.DistantLight.Define(stage, "/World/ConceptualFillPair")
fill.CreateIntensityAttr(3500.0); fill.CreateAngleAttr(0.35); fill.CreateColorAttr((0.75, 0.82, 1.0))
carb.settings.get_settings().set_float("/rtx/post/tonemap/exposure", 0.5)

# One close contact camera for both panels: wrist, peg, hole and fixture remain
# in frame while the lateral change is visible between the two states.
camera_target = np.array([0.50, 0.0, 0.17], dtype=np.float32)
camera_position = np.array([1.02, -0.78, 0.50], dtype=np.float32)
camera = rep.create.camera(position=tuple(camera_position), look_at=tuple(camera_target), focal_length=40.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0))
product = rep.create.render_product(camera, resolution=(1920, 1080))
ann = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
ann.attach(product if isinstance(product, str) else product.path)

robot = stage.GetPrimAtPath("/World/envs/env_0/Robot")
held = stage.GetPrimAtPath("/World/envs/env_0/HeldAsset")
if not robot.IsValid() or not held.IsValid(): raise RuntimeError("Factory Robot/HeldAsset roots not found")
robot_op = UsdGeom.Xformable(robot).AddTranslateOp(opSuffix="conceptual_pair")
held_op = UsdGeom.Xformable(held).AddTranslateOp(opSuffix="conceptual_pair")

states = {
    "02_contact_misalignment": np.array([0.020, 0.010, 0.004], dtype=np.float32),
    "03_lateral_recovery": np.array([0.004, 0.002, 0.001], dtype=np.float32),
}
out = a.output / "contact_recovery_pair"; out.mkdir(parents=True, exist_ok=True)
for name, delta in states.items():
    value = Gf.Vec3d(*[float(x) for x in delta])
    for _ in range(7):
        robot_op.Set(value); held_op.Set(value); app.update(); env.sim.render(); robot_op.Set(value); held_op.Set(value)
    env.sim.render()
    pixels = np.asarray(ann.get_data())[..., :3].astype(np.uint8)
    if pixels.shape != (1080, 1920, 3) or float(pixels.std()) <= 1.0: raise RuntimeError(f"invalid render {name}: {pixels.shape}, {pixels.std()}")
    Image.fromarray(pixels, mode="RGB").save(out / f"{name}.png", format="PNG")

metadata = {
    "case_type": "simulator_rendered_conceptual_illustration_pair",
    "formal64_pair_id": None, "is_formal64_evaluation_trajectory": False, "is_policy_output": False,
    "scene_seed": 20260819, "camera_pose": camera_position.tolist(), "camera_target": camera_target.tolist(),
    "resolution": [1920, 1080],
    "attachment_method": "identical USD translation on /World/envs/env_0/Robot and /World/envs/env_0/HeldAsset",
    "interpretation": "02 shows a larger lateral peg-hole error; 03 shows the same scene from the same camera after the scripted lateral offset is reduced.",
    "states": {name: {"png": f"contact_recovery_pair/{name}.png", "waypoint_delta_xyz_m": delta.tolist()} for name, delta in states.items()},
    "caption": "Simulator-rendered conceptual illustration of lateral contact-stage recovery. These scripted states are not Formal64 evaluation trajectories or quantitative outcomes.",
}
(a.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
env_obj.close(); app.close(); print(json.dumps({"output": str(a.output), "states": list(states), "resolution": [1920, 1080]}))
