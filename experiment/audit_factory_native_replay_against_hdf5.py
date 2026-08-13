#!/usr/bin/env python3
"""Replay a raw demo to its first trained policy label and compare observations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--hdf5',type=Path,required=True);p.add_argument('--demo',default='demo_00000');p.add_argument('--output',type=Path,required=True);p.add_argument('--resolution',type=int,default=84);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import h5py,gymnasium as gym,numpy as np,omni.replicator.core as rep,torch
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
def npv(x): return x.detach().float().cpu().numpy()
def state(e): return np.concatenate((npv(e.joint_pos[0,:9]),npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)
def action_to(e,o,z,clip=.35):
 t=npv(e.fixed_pos[0])+np.array([o[0],o[1],z],np.float32);u=np.zeros(6,np.float32);u[:3]=np.clip((t-npv(e.held_pos[0]))/.01,-clip,clip);return torch.from_numpy(u).to(e.device)[None]
def cams():
 out={}
 for n,pos in {'rgb_table':(1.10,0,.80),'rgb_side':(.65,-.85,.52)}.items():
  c=rep.create.camera(position=pos,look_at=(0,0,.30));q=rep.create.render_product(c,resolution=(a.resolution,a.resolution));x=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');x.attach(q if isinstance(q,str) else q.path);out[n]=x
 return out
def img(q):
 x=np.asarray(q.get_data());return (x[...,:3] if x.shape[-1]==4 else x).astype(np.uint8)
with h5py.File(a.hdf5,'r') as f:
 g=f['demos'][a.demo];seed=int(g.attrs['episode_seed']);sector=int(g.attrs['direction_sector']);mag=float(g.attrs['contact_offset_command_m']);ref=int(np.flatnonzero(np.asarray(g['is_policy_label']).reshape(-1))[0]);rs=np.asarray(g['state'][ref]);rt=np.asarray(g['joint_torque'][ref-29:ref+1]);ri={n:np.asarray(g[n][ref])[...,:3] for n in ('rgb_table','rgb_side')};ra=np.asarray(g['action'][ref])
cfg=parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0',device='cuda:0',num_envs=1);cfg.seed=seed;cfg.episode_length_s=40;cfg.task.hand_init_pos_noise=[.006,.006,.004];cfg.task.held_asset_pos_noise=[.0025,.0025,.001];cfg.sim.render_interval=cfg.decimation
env=gym.make('Isaac-Factory-PegInsert-Direct-v0',cfg=cfg);e=env.unwrapped;ans=cams();env.reset(seed=seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
for _ in range(3):env.step(torch.zeros((1,6),dtype=torch.float32,device=e.device));e.sim.render()
o=np.array([np.cos(2*np.pi*sector/8),np.sin(2*np.pi*sector/8)],np.float32)*mag;phase='lift';ps=0;hist=[];got={}
for t in range(ref+1):
 if phase=='lift':u=action_to(e,np.zeros(2,np.float32),.030);phase,ps=('offset',0) if ps>=24 else (phase,ps)
 elif phase=='offset':u=action_to(e,o,.030);phase,ps=('approach',0) if ps>=23 else (phase,ps)
 elif phase=='approach':u=action_to(e,o,max(-.002,.025-.00027*ps));phase,ps=('history',0) if ps>=99 else (phase,ps)
 elif phase=='history':u=action_to(e,o,-.002,clip=.20);hist.append(npv(e.joint_torque[0,:7]).astype(np.float32));phase,ps=('unload',0) if ps>=30 else (phase,ps)
 else:u=action_to(e,o,.012)
 if t==ref:got={'state':state(e),'torque':np.stack((hist+[npv(e.joint_torque[0,:7]).astype(np.float32)])[-30:]),'images':{n:img(q) for n,q in ans.items()},'action':npv(u[0]),'phase':phase};break
 env.step(u);e.sim.render();ps+=1
def met(x,y):
 a=x.astype(np.float64);b=y.astype(np.float64);d=a-b;return {'max_abs':float(np.max(np.abs(d))),'mean_abs':float(np.mean(np.abs(d))),'rmse':float(np.sqrt(np.mean(d**2)))}
r={'demo':a.demo,'seed':seed,'sector':sector,'reference_frame':ref,'replay_phase_at_reference':got['phase'],'state':met(got['state'],rs),'torque_window':met(got['torque'],rt),'oracle_action':met(got['action'],ra),'images':{n:met(got['images'][n],ri[n]) for n in ri},'reference_action_first3':ra[:3].tolist(),'replay_action_first3':got['action'][:3].tolist()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print('[NATIVE_REPLAY_AUDIT]',json.dumps(r),flush=True)
for n in ri:np.save(a.output.parent/f'{a.demo}_{n}_reference.npy',ri[n]);np.save(a.output.parent/f'{a.demo}_{n}_replay.npy',got['images'][n])
env.close();app.close()
