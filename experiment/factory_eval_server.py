#!/usr/bin/env python3
"""Isaac-only Factory peg-in-hole evaluation server (loopback pickle protocol)."""
from __future__ import annotations
import argparse,json,pickle,socket,struct
from collections import deque
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=5588); p.add_argument('--episodes',type=int,default=10); p.add_argument('--seed',type=int,default=3100); p.add_argument('--max-steps',type=int,default=360); p.add_argument('--output',type=Path,required=True); AppLauncher.add_app_launcher_args(p); a=p.parse_args(); app=AppLauncher(a).app
import gymnasium as gym
import numpy as np
import omni.replicator.core as rep
import torch
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

def send(c,x):
 b=pickle.dumps(x,protocol=5); c.sendall(struct.pack('!Q',len(b))+b)
def recv(c):
 h=b''
 while len(h)<8:
  z=c.recv(8-len(h));
  if not z: raise ConnectionError('client disconnected')
  h+=z
 n=struct.unpack('!Q',h)[0]; b=b''
 while len(b)<n:
  z=c.recv(min(1<<20,n-len(b)))
  if not z: raise ConnectionError('client disconnected')
  b+=z
 return pickle.loads(b)
def wait_action(c):
 while True:
  try: return recv(c)
  except socket.timeout: app.update()
def npv(x): return x.detach().float().cpu().numpy()
def state(e): return np.concatenate((npv(e.joint_pos[0,:9]),npv(e.fingertip_midpoint_pos[0]))).astype(np.float32)
def strict(e):
 h,f=npv(e.held_pos[0]),npv(e.fixed_pos[0]); xy=float(np.linalg.norm(h[:2]-f[:2])); z=float(h[2]-f[2]); return xy<.0025 and z<.001,xy,z
def cameras():
 out={}
 for name,pos in {'camera1':(1.10,0,.80),'camera2':(.65,-.85,.52)}.items():
  cam=rep.create.camera(position=pos,look_at=(0,0,.30)); prod=rep.create.render_product(cam,resolution=(84,84)); prod=prod if isinstance(prod,str) else prod.path; an=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu'); an.attach(prod); out[name]=an
 return out
def frame(an):
 x=np.asarray(an.get_data()); return x[...,:3].astype(np.uint8) if x.shape[-1]==4 else x.astype(np.uint8)
cfg=parse_env_cfg('Isaac-Factory-PegInsert-Direct-v0',device='cuda:0',num_envs=1); cfg.sim.render_interval=cfg.decimation; env=gym.make('Isaac-Factory-PegInsert-Direct-v0',cfg=cfg); e=env.unwrapped; anns=cameras()
s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',a.port)); s.listen(1); print(f'[FACTORY_SERVER] listening port={a.port}',flush=True); c,_=s.accept(); print('[FACTORY_SERVER] client_connected',flush=True); c.settimeout(0.25)
rows=[]
try:
 for ep in range(a.episodes):
  env.reset(seed=a.seed+ep); print(f'[FACTORY_SERVER] episode={ep} prewarm',flush=True)
  env.step(torch.zeros((1,6),dtype=torch.float32,device=e.device)); e.sim.render()
  ts=deque(maxlen=30); hit=False; xy=z=float('inf')
  for step in range(a.max_steps):
   e.sim.render(); t=np.array([float(np.linalg.norm(npv(e.joint_torque[0,:7])))],np.float32); ts.append(t)
   while len(ts)<30: ts.appendleft(t.copy())
   send(c,{'kind':'obs','state':state(e),'camera1':frame(anns['camera1']),'camera2':frame(anns['camera2']),'torque':np.stack(ts)})
   msg=wait_action(c)
   if msg.get('kind')!='action': raise ValueError(msg)
   u=np.asarray(msg['action'],np.float32)
   if u.shape!=(6,): raise ValueError(f'action {u.shape}')
   print(f'[FACTORY_SERVER] step={step} action={u.tolist()}',flush=True)
   env.step(torch.from_numpy(u)[None].to(e.device)); print(f'[FACTORY_SERVER] step={step} physics_done',flush=True); hit,xy,z=strict(e)
   if hit: break
  row={'episode':ep,'seed':a.seed+ep,'success':bool(hit),'steps':step+1,'final_xy_error_m':xy,'final_depth_m':z}; rows.append(row); send(c,{'kind':'episode',**row}); print('[FACTORY_SERVER]',row,flush=True)
 send(c,{'kind':'done','rows':rows}); summary={'episodes':a.episodes,'successes':sum(x['success'] for x in rows),'success_rate':sum(x['success'] for x in rows)/a.episodes,'strict_definition':'xy<0.0025m and held_z-hole_z<0.001m','rows':rows}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(summary,indent=2)+'\n'); print('[FACTORY_SERVER] SUMMARY',json.dumps(summary),flush=True)
finally:
 c.close(); s.close(); env.close(); app.close()
