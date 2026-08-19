#!/usr/bin/env python3
"""Interactive Isaac Sim Factory peg-in-hole scene for manual screenshot capture.

Launch with LIVESTREAM=2 (WebRTC private mode) and connect one Isaac Sim
WebRTC client. This is a scripted conceptual scene, not a policy evaluation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--snapshot',type=Path,required=True)
p.add_argument('--state',choices=['initial','contact','recovery','aligned'],default='contact')
AppLauncher.add_app_launcher_args(p); a=p.parse_args()
app=AppLauncher(a).app
import carb
carb.settings.get_settings().set_bool('/physics/fabricUpdateTransformations',True)
import gymnasium as gym
import omni.usd
import numpy as np
import torch
from pxr import Gf, UsdLux, UsdGeom
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def restore(e,s):
    e._robot.write_root_pose_to_sim(s['robot_root'][:,:7]); e._robot.write_root_velocity_to_sim(s['robot_root'][:,7:]); e._robot.write_joint_state_to_sim(s['robot_joint_pos'],s['robot_joint_vel'])
    e._held_asset.write_root_pose_to_sim(s['held_root'][:,:7]); e._held_asset.write_root_velocity_to_sim(s['held_root'][:,7:]); e._fixed_asset.write_root_pose_to_sim(s['fixed_root'][:,:7]); e._fixed_asset.write_root_velocity_to_sim(s['fixed_root'][:,7:])
    if 'ctrl_target_joint_pos' in s: e._robot.set_joint_position_target(s['ctrl_target_joint_pos'])
    e.scene.write_data_to_sim(); e.scene.update(dt=e.physics_dt); e.sim.render()
    for _ in range(20): app.update(); e.sim.render()

cfg=parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0',device='cuda:0',num_envs=1)
cfg.seed=20260819; cfg.episode_length_s=120; cfg.task.hand_init_pos_noise=[0.0,0.0,0.0]; cfg.task.held_asset_pos_noise=[0.0,0.0,0.0]; cfg.sim.render_interval=cfg.decimation; cfg.sim.use_fabric=True
env=gym.make('Isaac-Factory-PegInsert-Direct-v0',cfg=cfg); e=env.unwrapped; env.reset(seed=20260819)
s={k:(v.to('cuda') if torch.is_tensor(v) else v) for k,v in torch.load(a.snapshot,map_location='cpu',weights_only=False).items()}; restore(e,s)
stage=omni.usd.get_context().get_stage()
dome=UsdLux.DomeLight.Define(stage,'/World/InteractiveDome'); dome.CreateIntensityAttr(1800.0); dome.CreateColorAttr((0.82,0.86,0.92))
key=UsdLux.SphereLight.Define(stage,'/World/InteractiveKey'); key.CreateIntensityAttr(9000.0); key.CreateRadiusAttr(0.12); UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(0.92,-0.55,1.18))
fill=UsdLux.DistantLight.Define(stage,'/World/InteractiveFill'); fill.CreateIntensityAttr(3500.0); fill.CreateAngleAttr(0.35); fill.CreateColorAttr((0.75,0.82,1.0))
carb.settings.get_settings().set_float('/rtx/post/tonemap/exposure',0.5)
waypoints={'initial':np.array([0.012,0.007,0.009],np.float32),'contact':np.array([0.006,0.003,0.002],np.float32),'recovery':np.array([0.0015,0.0007,0.0],np.float32),'aligned':np.array([0.0,0.0,-0.003],np.float32)}
d=waypoints[a.state]; br=e._robot.data.root_state_w[0].detach().clone(); bh=e._held_asset.data.root_state_w[0].detach().clone(); br[:3]+=torch.tensor(d,device='cuda'); bh[:3]+=torch.tensor(d,device='cuda'); e._robot.write_root_pose_to_sim(br[None,:7]); e._held_asset.write_root_pose_to_sim(bh[None,:7]); e.scene.write_data_to_sim(); e.scene.update(dt=e.physics_dt)
# Isaac Lab's active viewport camera is controllable from the WebRTC client.
e.sim.set_camera_view(eye=[0.82,-0.45,0.32],target=[0.56,0.0,0.115])
print({'state':a.state,'livestream':'LIVESTREAM=2','camera_eye':[0.82,-0.45,0.32],'camera_target':[0.56,0.0,0.115]},flush=True)
while app.is_running():
    app.update(); e.sim.render()
env.close(); app.close()
