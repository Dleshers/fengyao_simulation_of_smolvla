#!/usr/bin/env python3
"""Render a close, contact-stage pair using a bounded rim-interference state.

The peg centre is placed 6 mm from the hole centre (8 mm peg/hole geometry),
so the peg overlaps the rim by about 2 mm while remaining at the mouth rather
than being far away. This is a scripted geometric contact-stage illustration;
the metadata records the contact proxy explicitly and does not claim a policy
or a force trace.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(); p.add_argument("--snapshot", type=Path, required=True); p.add_argument("--output", type=Path, required=True); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app
import carb
carb.settings.get_settings().set_bool("/isaaclab/render/offscreen", True)
carb.settings.get_settings().set_bool("/physics/fabricUpdateTransformations", True)
import gymnasium as gym, numpy as np, omni.replicator.core as rep, omni.usd, torch
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg=parse_env_cfg("Isaac-Factory-PegInsert-Direct-v0",device="cuda:0",num_envs=1); cfg.seed=20260820; cfg.episode_length_s=30; cfg.task.hand_init_pos_noise=[0.,0.,0.]; cfg.task.held_asset_pos_noise=[0.,0.,0.]; cfg.sim.render_interval=cfg.decimation; cfg.sim.use_fabric=False
env_obj=gym.make("Isaac-Factory-PegInsert-Direct-v0",cfg=cfg); env=env_obj.unwrapped; env_obj.reset(seed=20260820)
s=torch.load(a.snapshot,map_location="cpu",weights_only=False); s={k:(v.to("cuda") if torch.is_tensor(v) else v) for k,v in s.items()}
env._robot.write_root_pose_to_sim(s["robot_root"][:,:7]); env._robot.write_root_velocity_to_sim(s["robot_root"][:,7:]); env._robot.write_joint_state_to_sim(s["robot_joint_pos"],s["robot_joint_vel"]); env._held_asset.write_root_pose_to_sim(s["held_root"][:,:7]); env._held_asset.write_root_velocity_to_sim(s["held_root"][:,7:]); env._fixed_asset.write_root_pose_to_sim(s["fixed_root"][:,:7]); env._fixed_asset.write_root_velocity_to_sim(s["fixed_root"][:,7:]); env.scene.write_data_to_sim(); env.scene.update(dt=env.physics_dt); env._compute_intermediate_values(dt=env.physics_dt); env.sim.render()
base_joint=s["robot_joint_pos"].clone(); base_held=s["held_root"].clone(); base_fixed=s["fixed_root"].clone(); base_tip=env.fingertip_midpoint_pos.clone()
stage=omni.usd.get_context().get_stage(); dome=UsdLux.DomeLight.Define(stage,"/World/ConceptualDomeJoint"); dome.CreateIntensityAttr(1800.); dome.CreateColorAttr((.82,.86,.92)); key=UsdLux.SphereLight.Define(stage,"/World/ConceptualKeyJoint"); key.CreateIntensityAttr(9000.); key.CreateRadiusAttr(.12); UsdGeom.Xformable(key).AddTranslateOp().Set(Gf.Vec3d(.92,-.55,1.18)); fill=UsdLux.DistantLight.Define(stage,"/World/ConceptualFillJoint"); fill.CreateIntensityAttr(3500.); fill.CreateAngleAttr(.35); fill.CreateColorAttr((.75,.82,1.)); carb.settings.get_settings().set_float("/rtx/post/tonemap/exposure",.5)
target=base_fixed[0,:3].detach().cpu().numpy()+np.array([0.,0.,.025],dtype=np.float32); position=target+np.array([.20,-.26,.16],dtype=np.float32); camera=rep.create.camera(position=tuple(position),look_at=tuple(target),focal_length=55.,horizontal_aperture=20.955,clipping_range=(.01,100.)); product=rep.create.render_product(camera,resolution=(1920,1080)); ann=rep.AnnotatorRegistry.get_annotator("rgb",device="cpu"); ann.attach(product if isinstance(product,str) else product.path)
states={"02_contact_misalignment":0.010,"03_lateral_recovery":0.0}; out=a.output/"contact_recovery_pair"; out.mkdir(parents=True,exist_ok=True); measured={}; contact_offset=torch.tensor([0.006,0.0],device=base_held.device,dtype=base_held.dtype)
for name,dq in states.items():
    joints=base_joint.clone(); joints[:,0]+=dq; env._robot.write_joint_state_to_sim(joints,torch.zeros_like(s["robot_joint_vel"])); env.scene.write_data_to_sim(); env.scene.update(dt=env.physics_dt); env._compute_intermediate_values(dt=env.physics_dt); shift=(env.fingertip_midpoint_pos-base_tip).detach().clone(); held=base_held.clone(); held[:,:3]+=shift
    if name == "02_contact_misalignment": held[:,:2] = base_fixed[:,:2] + contact_offset.view(1,2)
    else: held[:,:2] = base_fixed[:,:2]
    env._held_asset.write_root_pose_to_sim(held[:,:7]); env._held_asset.write_root_velocity_to_sim(torch.zeros_like(held[:,7:])); env.scene.write_data_to_sim(); measured[name] = {"fingertip_shift_m": shift[0].detach().cpu().tolist(), "peg_center_offset_from_hole_m": (held[0,:2]-base_fixed[0,:2]).detach().cpu().tolist()}
    for _ in range(6): app.update(); env.sim.render()
    pixels=np.asarray(ann.get_data())[...,:3].astype(np.uint8)
    if pixels.shape!=(1080,1920,3) or float(pixels.std())<=1.: raise RuntimeError(f"invalid {name}: {pixels.shape}")
    Image.fromarray(pixels,mode="RGB").save(out/f"{name}.png",format="PNG")
metadata={"case_type":"simulator_rendered_contact_stage_geometric_pair","formal64_pair_id":None,"is_formal64_evaluation_trajectory":False,"is_policy_output":False,"camera_pose":position.tolist(),"camera_target":target.tolist(),"resolution":[1920,1080],"geometry":{"peg_diameter_m":0.008,"hole_diameter_m":0.008,"contact_stage_center_offset_m":0.006,"estimated_rim_interference_m":0.002,"contact_proxy":"peg/hole radial overlap at mouth; no force/contact trace claimed"},"joint_change":"Panda joint 1: +0.010 rad to 0 rad; held peg explicitly follows the 6 mm contact-stage offset","measured_states":measured,"caption":"Close simulator-rendered conceptual illustration: peg touches/intersects the hole rim before lateral recovery. Scripted stills are not Formal64 evaluation trajectories or quantitative outcomes."}
(a.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n"); env_obj.close(); app.close(); print(json.dumps(metadata))
