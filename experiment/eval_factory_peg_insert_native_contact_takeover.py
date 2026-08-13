#!/usr/bin/env python3
"""Native-reset physical-contact policy takeover evaluation (7D torque compatible)."""
from __future__ import annotations
import argparse, json, os, sys, tempfile
from collections import deque
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--policy-path',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--repo-id',required=True); p.add_argument('--output',type=Path,required=True)
p.add_argument('--coarse-policy-path',type=Path,default=None,help='Optional visual policy used until the first near-hole threshold crossing.')
p.add_argument('--coarse-until-xy-m',type=float,default=.0025,help='Latch from coarse policy to the evaluated policy once XY error is at or below this value.')
p.add_argument('--fine-xy-action-clip',type=float,default=0.0,help='Optional symmetric XY action clip after the fine-policy latch; 0 disables.')
p.add_argument('--episodes',type=int,default=8); p.add_argument('--max-attempts',type=int,default=64); p.add_argument('--seed',type=int,default=8260); p.add_argument('--max-steps',type=int,default=240)
p.add_argument('--n-action-steps',type=int,default=1,help='Number of cached actions executed before observing and replanning (use 1 for closed-loop evaluation).')
p.add_argument('--torque-mode',choices=('none','original','zero','shuffle'),default='none'); p.add_argument('--contact-offset-m',type=float,default=.006); p.add_argument('--contact-history-steps',type=int,default=30); p.add_argument('--min-contact-torque-delta',type=float,default=.03)
p.add_argument('--pre-takeover-unload-steps',type=int,default=0,help='Controller unload steps applied after contact history and before policy takeover. Use 15 for phase>=5 checkpoints to match collection.')
p.add_argument('--hand-noise-xy-m',type=float,default=.006); p.add_argument('--held-noise-xy-m',type=float,default=.0025)
p.add_argument('--inference-samples',type=int,default=1,help='Independent flow samples averaged per observation.')
p.add_argument('--deterministic-flow-noise',action='store_true',help='Use per-episode/step/sample local RNG so paired arms receive identical flow noise.')
p.add_argument('--flow-noise-seed',type=int,default=None,help='Base seed for deterministic flow noise; defaults to the episode seed.')
p.add_argument('--flow-noise-fixed-across-steps',action='store_true',help='Reuse each deterministic sample noise throughout a rollout for temporal consistency.')
p.add_argument('--action-clip',type=float,default=0.0,help='Symmetric action support clip; 0 disables clipping.')
p.add_argument('--save-traces',action='store_true',help='Store per-step action and signed pose traces.')
AppLauncher.add_app_launcher_args(p); a=p.parse_args()
if a.inference_samples < 1: p.error('--inference-samples must be positive')
if a.inference_samples > 1 and a.n_action_steps != 1: p.error('multi-sample averaging requires --n-action-steps 1')
if a.action_clip < 0: p.error('--action-clip must be non-negative')
if a.pre_takeover_unload_steps < 0: p.error('--pre-takeover-unload-steps must be non-negative')
if a.coarse_until_xy_m <= 0: p.error('--coarse-until-xy-m must be positive')
if a.fine_xy_action_clip < 0: p.error('--fine-xy-action-clip must be non-negative')
app=AppLauncher(a).app
site=os.environ.get('LEROBOT_SITE_PACKAGES'); source=os.environ.get('LEROBOT_SOURCE')
if site: sys.path.insert(0,site)
if source: sys.path.insert(0,source)
import importlib.machinery, types
boto3=types.ModuleType('boto3'); boto3.__spec__=importlib.machinery.ModuleSpec('boto3',loader=None); sys.modules['boto3']=boto3
import gymnasium as gym, numpy as np, omni.replicator.core as rep, torch
from PIL import Image
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy,make_pre_post_processors

def npv(x): return x.detach().float().cpu().numpy()
def state(e): return np.concatenate((npv(e.joint_pos[0,:9]),npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)
def pose(e):
 d=npv(e.held_pos[0])-npv(e.fixed_pos[0]); return float(np.linalg.norm(d[:2])),float(d[2])
def strict(e):
 xy,z=pose(e); return xy<.0025 and -.002<=z<=.001,xy,z
def anchor(e): return npv(e.fingertip_midpoint_pos[0])-npv(e.held_pos[0])
def action_to(e,off,z,clip=.35):
 target=npv(e.fixed_pos[0])+np.array([off[0],off[1],z],np.float32); u=np.zeros(6,np.float32); u[:3]=np.clip((target-npv(e.held_pos[0]))/.01,-clip,clip); return torch.from_numpy(u).to(e.device)[None]
def cams():
 out={}
 for name,pos in {'camera1':(1.10,0,.80),'camera2':(.65,-.85,.52)}.items():
  c=rep.create.camera(position=pos,look_at=(0,0,.30)); product=rep.create.render_product(c,resolution=(84,84)); ann=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu'); ann.attach(product if isinstance(product,str) else product.path); out[name]=ann
 return out
def image(ann):
 x=np.asarray(ann.get_data()); x=x[...,:3] if x.shape[-1]==4 else x
 return np.array(Image.fromarray(x.astype(np.uint8)).resize((224,224),Image.Resampling.BILINEAR),dtype=np.uint8,copy=True)
def camera_batch(anns):
 return {f'observation.images.{n}':torch.from_numpy(image(q)[None]).permute(0,3,1,2).float()/255. for n,q in anns.items()}

raw=json.loads((a.policy_path/'config.json').read_text()); raw.pop('tactile_token_mode',None); compat=Path(tempfile.mkdtemp(prefix='native_contact_cfg_')); (compat/'config.json').write_text(json.dumps(raw))
cfg=PreTrainedConfig.from_pretrained(compat); cfg.pretrained_path=str(a.policy_path); cfg.device='cuda'; cfg.n_action_steps=a.n_action_steps
if a.torque_mode=='none': cfg.use_torque_lstm=False
meta=LeRobotDatasetMetadata(a.repo_id,root=a.dataset_root); policy=make_policy(cfg=cfg,ds_meta=meta); policy.eval()
pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(a.policy_path),preprocessor_overrides={'device_processor':{'device':'cuda'}})
coarse_policy=coarse_pre=coarse_post=None
if a.coarse_policy_path is not None:
 craw=json.loads((a.coarse_policy_path/'config.json').read_text()); craw.pop('tactile_token_mode',None); ccompat=Path(tempfile.mkdtemp(prefix='native_coarse_cfg_')); (ccompat/'config.json').write_text(json.dumps(craw))
 ccfg=PreTrainedConfig.from_pretrained(ccompat); ccfg.pretrained_path=str(a.coarse_policy_path); ccfg.device='cuda'; ccfg.n_action_steps=a.n_action_steps; ccfg.use_torque_lstm=False
 coarse_policy=make_policy(cfg=ccfg,ds_meta=meta); coarse_policy.eval()
 coarse_pre,coarse_post=make_pre_post_processors(policy_cfg=ccfg,pretrained_path=str(a.coarse_policy_path),preprocessor_overrides={'device_processor':{'device':'cuda'}})
ecfg=parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0',device='cuda:0',num_envs=1); ecfg.seed=a.seed; ecfg.episode_length_s=40; ecfg.task.hand_init_pos_noise=[a.hand_noise_xy_m,a.hand_noise_xy_m,.004]; ecfg.task.held_asset_pos_noise=[a.held_noise_xy_m,a.held_noise_xy_m,.001]; ecfg.sim.render_interval=ecfg.decimation
env=gym.make('Isaac-Factory-PegInsert-Direct-v0',cfg=ecfg); e=env.unwrapped; anns=cams(); rng=np.random.default_rng(a.seed+999); rows=[]
for attempt in range(a.max_attempts):
 if len(rows)>=a.episodes: break
 seed=a.seed+attempt; sector=attempt%8; off=np.array([np.cos(2*np.pi*sector/8),np.sin(2*np.pi*sector/8)],np.float32)*a.contact_offset_m
 env.reset(seed=seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); policy.reset()
 if coarse_policy is not None: coarse_policy.reset()
 for _ in range(3): env.step(torch.zeros((1,6),device=e.device)); e.sim.render()
 phase='lift'; ps=0; hist=deque(maxlen=30); contact_xy=contact_z=delta=0.; start_anchor=anchor(e); valid=False; handover=None
 for warm in range(260):
  xy,z=pose(e)
  if phase=='lift': u=action_to(e,np.zeros(2,np.float32),.030); phase,ps=('offset',0) if ps>=24 else (phase,ps)
  elif phase=='offset': u=action_to(e,off,.030); phase,ps=('approach',0) if ps>=23 else (phase,ps)
  elif phase=='approach':
   u=action_to(e,off,max(-.002,.025-.00027*ps)); phase,ps=('history',0) if ps>=99 else (phase,ps)
  else:
   u=action_to(e,off,-.002,clip=.20); hist.append(npv(e.joint_torque[0,:7]).astype(np.float32))
   if ps<=1: contact_xy,contact_z=xy,z
   if len(hist)>=a.contact_history_steps:
    h=np.stack(hist); base=np.median(h[:10],axis=0); delta=float(np.max(np.linalg.norm(h-base,axis=1))); hit,_,_=strict(e); drift=float(np.linalg.norm(anchor(e)-start_anchor))
    valid=bool(.0025<=contact_xy<=.009 and .015<=contact_z<=.032 and delta>=a.min_contact_torque_delta and not hit and drift<=.003)
    env.step(u); e.sim.render(); handover=warm; break
  env.step(u); e.sim.render(); ps+=1
 if not valid:
  print('[NATIVE_TAKEOVER] reject',{'attempt':attempt,'seed':seed,'xy':contact_xy,'z':contact_z,'torque_delta':delta},flush=True); continue
 # The phase>=5 training view starts after the collector's deterministic
 # unload phase. Preserve the chronological torque window by recording each
 # pre-action sample exactly as the collector does.
 for _ in range(a.pre_takeover_unload_steps):
  hist.append(npv(e.joint_torque[0,:7]).astype(np.float32))
  env.step(action_to(e,off,.012)); e.sim.render()
 initial_xy,initial_z=pose(e); torques=deque(hist,maxlen=30); torque_trace=[]; action_trace=[]; pose_trace=[]; first=None; success=False
 fine_takeover=coarse_policy is None; takeover_step=0 if fine_takeover else None
 for step in range(a.max_steps):
  e.sim.render(); tq=npv(e.joint_torque[0,:7]).astype(np.float32); torques.append(tq); torque_trace.append(float(np.linalg.norm(tq)))
  current_xy,_=pose(e)
  if not fine_takeover and current_xy <= a.coarse_until_xy_m: fine_takeover=True; takeover_step=step
  batch={'observation.state':torch.from_numpy(state(e)[None]),'task':['Insert the peg into the hole']}; batch.update(camera_batch(anns))
  if a.torque_mode!='none':
   w=np.stack(torques)
   if a.torque_mode=='zero': w=np.zeros_like(w)
   elif a.torque_mode=='shuffle': w=w[rng.permutation(30)]
   batch['observation.gripper_torque']=torch.from_numpy(w[None])
  sampled=[]
  active_policy,active_pre,active_post=(policy,pre,post) if fine_takeover else (coarse_policy,coarse_pre,coarse_post)
  active_batch=batch if fine_takeover else {k:v for k,v in batch.items() if k!='observation.gripper_torque'}
  for sample_idx in range(a.inference_samples):
   noise=None
   if a.deterministic_flow_noise:
    generator=torch.Generator(device=e.device); generator.manual_seed((a.flow_noise_seed if a.flow_noise_seed is not None else seed) + sample_idx + (0 if a.flow_noise_fixed_across_steps else step * 1009))
    active_cfg=active_policy.config
    noise=torch.randn((1,active_cfg.chunk_size,active_cfg.max_action_dim),generator=generator,device=e.device)
   with torch.inference_mode(): sampled.append(active_post(active_policy.select_action(active_pre(active_batch),noise=noise)).detach().float().cpu().numpy())
  act=np.mean(sampled,axis=0)
  if a.action_clip > 0: act=np.clip(act,-a.action_clip,a.action_clip)
  if fine_takeover and a.fine_xy_action_clip > 0: act[:,:2]=np.clip(act[:,:2],-a.fine_xy_action_clip,a.fine_xy_action_clip)
  action_trace.append(act[0].tolist())
  env.step(torch.from_numpy(act).to(e.device)); success,xy,z=strict(e)
  d=npv(e.held_pos[0])-npv(e.fixed_pos[0]); pose_trace.append((xy,z,float(d[0]),float(d[1])))
  if first is None and xy<.0025: first=step+1
  if success: break
 aa=np.asarray(action_trace); pp=np.asarray(pose_trace); row={'episode':len(rows),'attempt':attempt,'seed':seed,'sector':sector,'initial_xy_error_m':initial_xy,'initial_depth_m':initial_z,'contact_xy_error_m':contact_xy,'contact_depth_m':contact_z,'contact_torque_delta':delta,'contact_history_frames':30,'pre_takeover_unload_steps':a.pre_takeover_unload_steps,'coarse_policy_path':str(a.coarse_policy_path) if a.coarse_policy_path else None,'coarse_until_xy_m':a.coarse_until_xy_m,'fine_takeover_step':takeover_step,'fine_xy_action_clip':a.fine_xy_action_clip,'deterministic_flow_noise':a.deterministic_flow_noise,'flow_noise_seed':a.flow_noise_seed,'flow_noise_fixed_across_steps':a.flow_noise_fixed_across_steps,'valid_non_success_initialization':True,'first_aligned_step':first,'success':bool(success),'steps':step+1,'min_xy_error_m':float(pp[:,0].min()),'min_depth_m':float(pp[:,1].min()),'final_xy_error_m':xy,'final_depth_m':z,'first_action':action_trace[0],'mean_action':aa.mean(0).tolist(),'mean_abs_action':np.abs(aa).mean(0).tolist(),'max_abs_action':np.abs(aa).max(0).tolist(),'mean_torque_norm':float(np.mean(torque_trace)),'policy_torque_mode':a.torque_mode,'inference_samples':a.inference_samples,'action_clip':a.action_clip}
 if a.save_traces: row.update(action_trace=action_trace,pose_trace_xy_z_dx_dy=pp.tolist())
 rows.append(row); print('[NATIVE_TAKEOVER]',row,flush=True)
valid=rows; aligned=[r for r in valid if r['first_aligned_step'] is not None]; succ=[r for r in valid if r['success']]
summary={'benchmark':'native_reset_physical_contact_takeover_v1','interpretation':'No pose write after reset. A controller creates off-centre physical rim contact and 30 chronological 7D torque frames, then optionally reproduces the collection unload phase before policy takeover.','policy':str(a.policy_path),'torque_mode':a.torque_mode,'n_action_steps':a.n_action_steps,'inference_samples':a.inference_samples,'action_clip':a.action_clip,'pre_takeover_unload_steps':a.pre_takeover_unload_steps,'coarse_policy_path':str(a.coarse_policy_path) if a.coarse_policy_path else None,'coarse_until_xy_m':a.coarse_until_xy_m,'fine_xy_action_clip':a.fine_xy_action_clip,'deterministic_flow_noise':a.deterministic_flow_noise,'flow_noise_seed':a.flow_noise_seed,'flow_noise_fixed_across_steps':a.flow_noise_fixed_across_steps,'episodes_requested':a.episodes,'episodes':len(rows),'valid_initializations':len(valid),'alignment_recoveries':len(aligned),'strict_recoveries':len(succ),'alignment_recovery_rate':len(aligned)/len(valid) if valid else None,'strict_recovery_rate':len(succ)/len(valid) if valid else None,'rows':rows}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(summary,indent=2)+'\n'); print('[NATIVE_TAKEOVER] SUMMARY',json.dumps(summary),flush=True)
env.close(); app.close()
