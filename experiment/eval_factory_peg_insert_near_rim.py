#!/usr/bin/env python3
"""Strict Factory peg-in-hole evaluation with near-rim recovery stratification."""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from collections import deque
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--policy-path',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--repo-id',required=True)
p.add_argument('--output',type=Path,required=True); p.add_argument('--episodes',type=int,default=10); p.add_argument('--seed',type=int,default=4100); p.add_argument('--max-steps',type=int,default=360); p.add_argument('--torque-mode',choices=('none','original','zero','shuffle'),default='none'); p.add_argument('--n-action-steps',type=int,default=1)
p.add_argument('--near-xy-threshold',type=float,default=.005,help='Near-rim lateral threshold in metres.')
p.add_argument('--near-depth-max',type=float,default=.005,help='Require held_z-hole_z <= this value for a near-rim event.')
AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app

site=os.environ.get('LEROBOT_SITE_PACKAGES'); source=os.environ.get('LEROBOT_SOURCE')
if site: sys.path.insert(0,site)
if source: sys.path.insert(0,source)
for key in list(sys.modules):
 if key == 'botocore' or key.startswith('botocore.'): del sys.modules[key]
import types, importlib.machinery
_boto3=types.ModuleType('boto3'); _boto3.__spec__=importlib.machinery.ModuleSpec('boto3',loader=None); sys.modules['boto3']=_boto3
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
from PIL import Image
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors

def npv(x): return x.detach().float().cpu().numpy()
def state(e): return np.concatenate((npv(e.joint_pos[0,:9]),npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)
def pose_error(e):
 h,f=npv(e.held_pos[0]),npv(e.fixed_pos[0]); return float(np.linalg.norm(h[:2]-f[:2])),float(h[2]-f[2])
def strict(e):
 xy,z=pose_error(e); return xy<.0025 and z<.001,xy,z
def img(ann):
 x=np.asarray(ann.get_data()); x=x[...,:3] if x.shape[-1]==4 else x
 return np.array(Image.fromarray(x.astype(np.uint8)).resize((224,224),Image.Resampling.BILINEAR),np.uint8,copy=True)
def cameras():
 out={}
 for name,pos in {'camera1':(1.10,0,.80),'camera2':(.65,-.85,.52)}.items():
  cam=rep.create.camera(position=pos,look_at=(0,0,.30)); prod=rep.create.render_product(cam,resolution=(84,84)); prod=prod if isinstance(prod,str) else prod.path
  an=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu'); an.attach(prod); out[name]=an
 return out

raw_cfg=json.loads((a.policy_path/'config.json').read_text()); raw_cfg.pop('tactile_token_mode',None)
compat_dir=Path(tempfile.mkdtemp(prefix='factory_near_rim_cfg_')); (compat_dir/'config.json').write_text(json.dumps(raw_cfg))
cfg=PreTrainedConfig.from_pretrained(compat_dir); cfg.pretrained_path=str(a.policy_path); cfg.device='cuda'; cfg.n_action_steps=a.n_action_steps
if a.torque_mode=='none': cfg.use_torque_lstm=False
meta=LeRobotDatasetMetadata(a.repo_id,root=a.dataset_root); policy=make_policy(cfg=cfg,ds_meta=meta); policy.eval()
pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(a.policy_path),preprocessor_overrides={'device_processor':{'device':'cuda'}})
ecfg=parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0',device='cuda:0',num_envs=1); ecfg.sim.render_interval=ecfg.decimation
env=gym.make('Isaac-Factory-PegInsert-Direct-v0',cfg=ecfg); e=env.unwrapped; anns=cameras(); a.output.parent.mkdir(parents=True,exist_ok=True)
rows=[]; rng=np.random.default_rng(a.seed)
for ep in range(a.episodes):
 episode_seed=a.seed+ep; env.reset(seed=episode_seed); torch.manual_seed(episode_seed); torch.cuda.manual_seed_all(episode_seed); policy.reset(); torques=deque(maxlen=30)
 hit=False; xy=z=float('inf'); metrics={"min_xy":float("inf"),"first_near":None,"first_nonterminal_near":None}
 def observe(step,cur_xy,cur_z,terminal):
  metrics["min_xy"]=min(metrics["min_xy"],cur_xy)
  if cur_xy<=a.near_xy_threshold and cur_z<=a.near_depth_max:
   if metrics["first_near"] is None: metrics["first_near"]=step
   if not terminal and metrics["first_nonterminal_near"] is None: metrics["first_nonterminal_near"]=step
 for step in range(a.max_steps):
  e.sim.render(); pre_xy,pre_z=pose_error(e); observe(step,pre_xy,pre_z,False)
  torque=np.array([float(np.linalg.norm(npv(e.joint_torque[0,:7])))],np.float32); torques.append(torque)
  while len(torques)<30: torques.appendleft(torque.copy())
  batch={'observation.state':torch.from_numpy(state(e)[None]),'observation.images.camera1':torch.from_numpy(img(anns['camera1'])[None]).permute(0,3,1,2).float()/255.,'observation.images.camera2':torch.from_numpy(img(anns['camera2'])[None]).permute(0,3,1,2).float()/255.,'task':['Insert the peg into the hole']}
  if a.torque_mode!='none':
   tw=np.stack(torques)
   if a.torque_mode=='zero': tw=np.zeros_like(tw)
   elif a.torque_mode=='shuffle': tw=tw[rng.permutation(len(tw))]
   batch['observation.gripper_torque']=torch.from_numpy(tw[None])
  with torch.inference_mode(): action=post(policy.select_action(pre(batch))).detach().float().cpu().numpy()
  env.step(torch.from_numpy(action).to(e.device)); hit,xy,z=strict(e); observe(step+1,xy,z,hit)
  if hit: break
 row={'episode':ep,'seed':episode_seed,'success':bool(hit),'steps':step+1,'final_xy_error_m':xy,'final_depth_m':z,'min_xy_error_m':metrics["min_xy"],'entered_near_rim':metrics["first_near"] is not None,'first_near_rim_step':metrics["first_near"],'near_rim_recovery_opportunity':metrics["first_nonterminal_near"] is not None,'first_nonterminal_near_rim_step':metrics["first_nonterminal_near"],'recovered_after_near_rim':bool(hit and metrics["first_nonterminal_near"] is not None)}
 rows.append(row); print('[NEAR_RIM_EVAL]',row,flush=True)
near=[r for r in rows if r['entered_near_rim']]; opp=[r for r in rows if r['near_rim_recovery_opportunity']]; recovered=[r for r in rows if r['recovered_after_near_rim']]
summary={'torque_mode':a.torque_mode,'n_action_steps':cfg.n_action_steps,'policy':str(a.policy_path),'episodes':a.episodes,'successes':sum(r['success'] for r in rows),'success_rate':sum(r['success'] for r in rows)/a.episodes,'strict_definition':'xy<0.0025m and held_z-hole_z<0.001m','near_rim_definition':f'xy<={a.near_xy_threshold}m and held_z-hole_z<={a.near_depth_max}m','near_rim_episodes':len(near),'near_rim_to_successes':sum(r['success'] for r in near),'near_rim_to_success_rate':(sum(r['success'] for r in near)/len(near)) if near else None,'recovery_opportunities':len(opp),'recoveries_after_nonterminal_near_rim':len(recovered),'near_rim_recovery_rate':(len(recovered)/len(opp)) if opp else None,'rows':rows}
a.output.write_text(json.dumps(summary,indent=2)+'\n'); print('[NEAR_RIM_EVAL] SUMMARY',json.dumps(summary),flush=True)
env.close(); app.close()
