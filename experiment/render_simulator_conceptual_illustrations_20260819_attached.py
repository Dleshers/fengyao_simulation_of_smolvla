#!/usr/bin/env python3
"""Render a fixed-camera conceptual contact-recovery sequence.

This revision deliberately applies the same USD translation to the complete
robot root and HeldAsset root.  The Factory robot has a fixed base, so writing
an articulation root pose does not move its rendered links; doing that while
moving HeldAsset alone makes the peg appear to float.  The shared visual
transform keeps wrist--peg attachment intact for these scripted illustrations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--snapshot", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

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


def settle_render(env, robot_op, held_op, delta, count=5):
    """Render without allowing a physics update to detach the visual pose."""
    value = Gf.Vec3d(float(delta[0]), float(delta[1]), float(delta[2]))
    for _ in range(count):
        robot_op.Set(value)
        held_op.Set(value)
        app.update()
        env.sim.render()
        # Isaac Sim may refresh articulation transforms during app.update().
        # Re-assert the paired root transform before the next render.
        robot_op.Set(value)
        held_op.Set(value)
    robot_op.Set(value)
    held_op.Set(value)
    env.sim.render()


def restore(env, snapshot):
    env._robot.write_root_pose_to_sim(snapshot["robot_root"][:, :7])
    env._robot.write_root_velocity_to_sim(snapshot["robot_root"][:, 7:])
    env._robot.write_joint_state_to_sim(snapshot["robot_joint_pos"], snapshot["robot_joint_vel"])
    env._held_asset.write_root_pose_to_sim(snapshot["held_root"][:, :7])
    env._held_asset.write_root_velocity_to_sim(snapshot["held_root"][:, 7:])
    env._fixed_asset.write_root_pose_to_sim(snapshot["fixed_root"][:, :7])
    env._fixed_asset.write_root_velocity_to_sim(snapshot["fixed_root"][:, 7:])
    if "ctrl_target_joint_pos" in snapshot:
        env._robot.set_joint_position_target(snapshot["ctrl_target_joint_pos"])
    env.scene.write_data_to_sim()
    env.scene.update(dt=env.physics_dt)
    env.sim.render()


cfg = parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0", device="cuda:0", num_envs=1)
cfg.seed = 20260819
cfg.episode_length_s = 30
cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]
cfg.sim.render_interval = cfg.decimation
cfg.sim.use_fabric = True
env_obj = gym.make("Isaac-Factory-PegInsert-Direct-v0", cfg=cfg)
env = env_obj.unwrapped
env_obj.reset(seed=20260819)
snapshot = {
    k: (v.to("cuda") if torch.is_tensor(v) else v)
    for k, v in torch.load(args.snapshot, map_location="cpu", weights_only=False).items()
}
restore(env, snapshot)

stage = omni.usd.get_context().get_stage()
dome = UsdLux.DomeLight.Define(stage, "/World/ConceptualDomeAttached")
dome.CreateIntensityAttr(1800.0)
dome.CreateColorAttr((0.82, 0.86, 0.92))
key = UsdLux.SphereLight.Define(stage, "/World/ConceptualKeyAttached")
key.CreateIntensityAttr(9000.0)
key.CreateRadiusAttr(0.12)
UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.92, -0.55, 1.18))
fill = UsdLux.DistantLight.Define(stage, "/World/ConceptualFillAttached")
fill.CreateIntensityAttr(3500.0)
fill.CreateAngleAttr(0.35)
fill.CreateColorAttr((0.75, 0.82, 1.0))
carb.settings.get_settings().set_float("/rtx/post/tonemap/exposure", 0.5)

# Wider, higher contact view: the wrist and held peg remain in frame even in
# the recovery/aligned panels, while the fixture and hole stay readable.
camera_target = np.array([0.42, 0.0, 0.20], dtype=np.float32)
camera_position = np.array([1.15, -0.95, 0.62], dtype=np.float32)
camera = rep.create.camera(
    position=tuple(camera_position.tolist()),
    look_at=tuple(camera_target.tolist()),
    focal_length=40.0,
    horizontal_aperture=20.955,
    clipping_range=(0.01, 100.0),
)
product = rep.create.render_product(camera, resolution=(1920, 1080))
annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
annotator.attach(product if isinstance(product, str) else product.path)

# The fixed-base robot cannot be translated through its articulation root.  A
# dedicated visual USD translation is therefore applied to both complete roots
# and reasserted before every capture.  Relative wrist-to-peg geometry is exact.
robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
held_prim = stage.GetPrimAtPath("/World/envs/env_0/HeldAsset")
if not robot_prim.IsValid() or not held_prim.IsValid():
    raise RuntimeError("Factory Robot/HeldAsset roots were not found")
robot_op = UsdGeom.Xformable(robot_prim).AddTranslateOp(opSuffix="conceptual_attachment")
held_op = UsdGeom.Xformable(held_prim).AddTranslateOp(opSuffix="conceptual_attachment")

waypoints = {
    "01_initial_lateral_error": np.array([0.025, 0.012, 0.012], dtype=np.float32),
    "02_contact_misalignment": np.array([0.012, 0.006, 0.004], dtype=np.float32),
    "03_lateral_recovery": np.array([0.004, 0.002, 0.001], dtype=np.float32),
    "04_aligned_insertion": np.array([0.000, 0.000, -0.004], dtype=np.float32),
}
out = args.output / "contact_recovery_sequence"
out.mkdir(parents=True, exist_ok=True)
for name, delta in waypoints.items():
    settle_render(env, robot_op, held_op, delta)
    pixels = np.asarray(annotator.get_data())[..., :3].astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[:2] != (1080, 1920):
        raise RuntimeError(f"unexpected render shape for {name}: {pixels.shape}")
    Image.fromarray(pixels, mode="RGB").save(out / f"{name}.png", format="PNG")

rendered = [np.asarray(Image.open(out / f"{name}.png"), dtype=np.float32) for name in waypoints]
validation = {"resolution": [1920, 1080], "frames": {}}
for name, frame in zip(waypoints, rendered):
    std = float(frame.std())
    validation["frames"][name] = {"nonuniform": bool(std > 1.0), "mean": float(frame.mean()), "std": std}
    if std <= 1.0:
        raise RuntimeError(f"blank conceptual render: {name}")
for index, name in enumerate(list(waypoints)[1:], start=1):
    validation["frames"][name]["mean_abs_diff_from_initial"] = float(np.abs(rendered[index] - rendered[0]).mean())
(args.output / "render_validation.json").write_text(json.dumps(validation, indent=2) + "\n")

metadata = {
    "case_type": "simulator_rendered_conceptual_illustration",
    "formal64_pair_id": None,
    "is_formal64_evaluation_trajectory": False,
    "is_policy_output": False,
    "scene_seed": 20260819,
    "script_or_waypoint_config": "experiment/render_simulator_conceptual_illustrations_20260819_attached.py",
    "source_scene": "Isaac Lab Factory PegInsert Direct scene",
    "camera_pose": camera_position.tolist(),
    "camera_target": camera_target.tolist(),
    "resolution": [1920, 1080],
    "attachment_preserved": True,
    "attachment_method": "identical USD translation on /World/envs/env_0/Robot and /World/envs/env_0/HeldAsset",
    "states": {
        name: {"step": i, "simulation_step": 0, "png": f"contact_recovery_sequence/{name}.png", "waypoint_delta_xyz_m": delta.tolist()}
        for i, (name, delta) in enumerate(waypoints.items())
    },
    "caption": "Simulator-rendered conceptual illustration of contact-stage recovery. The states are scripted illustrative configurations and are not Formal64 evaluation trajectories or quantitative outcomes.",
}
(out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
env_obj.close()
app.close()
print(json.dumps({"output": str(args.output), "states": list(waypoints), "resolution": [1920, 1080]}))
