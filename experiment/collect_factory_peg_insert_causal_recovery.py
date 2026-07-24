#!/usr/bin/env python3
"""Causal RGB/torque Factory peg-in-hole collection with recovery trajectories.

Only oracle recovery actions are added to the training frames.  Deliberate lateral
perturbation actions are executed but *not* labelled, so the policy never learns to
create the error it is expected to recover from.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--dataset-file',type=Path,required=True)
p.add_argument('--num-demos',type=int,default=120)
p.add_argument('--max-attempts',type=int,default=260)
p.add_argument('--max-steps',type=int,default=360)
p.add_argument('--resolution',type=int,default=84)
p.add_argument('--seed',type=int,default=20260724)
p.add_argument('--recovery-fraction',type=float,default=.5)
p.add_argument('--perturb-min-step',type=int,default=52)
p.add_argument('--perturb-max-step',type=int,default=96)
p.add_argument('--perturb-amplitude',type=float,default=.72)
AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app

import h5py, gymnasium as gym, numpy as np, omni.replicator.core as rep, torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def npv(x): return x.detach().float().cpu().numpy()
def state(e): return np.concatenate((npv(e.joint_pos[0,:9]),npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)
def strict(e):
 held,fixed=npv(e.held_pos[0]),npv(e.fixed_pos[0]); xy=float(np.linalg.norm(held[:2]-fixed[:2])); z=float(held[2]-fixed[2]); return xy<.0025 and z<.001,xy,z
def oracle(e,step):
 fixed,held=npv(e.fixed_pos[0]),npv(e.held_pos[0])
 if step<35: target=fixed+np.array([0,0,.030],np.float32)
 else:
  k=step-35; radius=min(.00035+.000035*k,.0045); theta=.24*k
  target=fixed+np.array([radius*np.cos(theta),radius*np.sin(theta),-.006],np.float32)
 u=np.zeros(6,np.float32); u[:3]=np.clip((target-held)/.01,-1,1); return torch.from_numpy(u).to(e.device)[None]
def cameras(res):
 out={}
 for name,pos in {'rgb_table':(1.10,0,.80),'rgb_side':(.65,-.85,.52)}.items():
  cam=rep.create.camera(position=pos,look_at=(0,0,.30)); prod=rep.create.render_product(cam,resolution=(res,res)); prod=prod if isinstance(prod,str) else prod.path
  ann=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu'); ann.attach(prod); out[name]=ann
 return out
def image(ann):
 x=np.asarray(ann.get_data()); x=x[...,:3] if x.shape[-1]==4 else x
 if x.ndim!=3 or x.shape[-1]!=3: raise RuntimeError(f'bad RGB {x.shape}')
 return x.astype(np.uint8,copy=False)

cfg=parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0',device='cuda:0',num_envs=1); cfg.seed=a.seed; cfg.sim.render_interval=cfg.decimation
env=gym.make('Isaac-Factory-PegInsert-Direct-v0',cfg=cfg); e=env.unwrapped; anns=cameras(a.resolution)
a.dataset_file.parent.mkdir(parents=True,exist_ok=True)
with h5py.File(a.dataset_file,'a') as f:
 f.attrs.update(format='factory_peg_insert_causal_recovery_v2',success_definition='held_vs_hole: xy<0.0025m and held_z-hole_z<0.001m',observation_action_alignment='state/rgb/joint_torque observed before action_t; action_t executed after recording',state_schema='proprio12:[joint_pos(9),fingertip_midpoint_pos(3)]; no peg/hole truth',action_schema='factory_delta_pose6:[xyz(3),axis_angle(3)]',force_schema='joint_torque7,controller_applied_wrench6',recovery_policy='unlabelled lateral perturbation followed by oracle-labelled recovery')
 demos=f.require_group('demos'); saved=len(demos); print(f'[CAUSAL_RECOVERY] resume={saved}/{a.num_demos} file={a.dataset_file}',flush=True)
 for attempt in range(saved+1,saved+a.max_attempts+1):
  if saved>=a.num_demos: break
  rng=np.random.default_rng(a.seed+attempt); env.reset(seed=a.seed+attempt)
  # Prime camera/sim; this zero action is deliberately not a training frame.
  env.step(torch.zeros((1,6),dtype=torch.float32,device=e.device)); e.sim.render()
  recovery=bool(rng.random()<a.recovery_fraction); perturb_step=int(rng.integers(a.perturb_min_step,a.perturb_max_step+1)) if recovery else -1
  b={k:[] for k in ('state','action','rgb_table','rgb_side','joint_torque','applied_wrench','is_recovery')}; ok=False; xy=z=float('inf'); injected=False
  for step in range(a.max_steps):
   if recovery and step==perturb_step:
    # Generate an off-nominal state without pairing this deliberately wrong action with an observation.
    kick=torch.zeros((1,6),dtype=torch.float32,device=e.device); kick[0,:2]=torch.from_numpy(rng.uniform(-a.perturb_amplitude,a.perturb_amplitude,2).astype(np.float32)).to(e.device)
    env.step(kick); e.sim.render(); injected=True; continue
   # Causal frame: all observation fields belong to the state before u_t is applied.
   u=oracle(e,step)
   b['state'].append(state(e)); b['action'].append(npv(u[0])); b['rgb_table'].append(image(anns['rgb_table'])); b['rgb_side'].append(image(anns['rgb_side'])); b['joint_torque'].append(npv(e.joint_torque[0,:7])); b['applied_wrench'].append(npv(e.applied_wrench[0])); b['is_recovery'].append(np.array([injected],np.uint8))
   env.step(u); e.sim.render(); ok,xy,z=strict(e)
   if ok: break
  if not ok:
   print(f'[CAUSAL_RECOVERY] reject attempt={attempt} recovery={recovery} xy={xy:.5f} z={z:.5f}',flush=True); continue
  g=demos.create_group(f'demo_{saved:05d}')
  for k,v in b.items(): g.create_dataset(k,data=np.stack(v),compression='gzip',compression_opts=1,shuffle=True)
  g.attrs.update(strict_success=True,success_step=step,final_xy_error_m=xy,final_depth_m=z,attempt=attempt,recovery_episode=recovery,perturb_step=perturb_step,perturb_executed=injected,frame_alignment='pre_action')
  f.flush(); saved+=1; print(f'[CAUSAL_RECOVERY] saved={saved}/{a.num_demos} attempt={attempt} recovery={recovery} frames={len(b["action"])} xy={xy:.5f} z={z:.5f}',flush=True)
env.close(); app.close()
if saved<a.num_demos: raise SystemExit(f'only {saved}/{a.num_demos} strict demos')
print(f'[CAUSAL_RECOVERY] complete strict_demos={saved}',flush=True)
