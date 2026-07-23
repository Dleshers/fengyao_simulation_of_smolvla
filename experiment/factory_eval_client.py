#!/usr/bin/env python3
"""LeRobot-only client for factory_eval_server.py."""
from __future__ import annotations
import argparse,json,pickle,socket,struct,tempfile,time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
p=argparse.ArgumentParser(); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=5588); p.add_argument('--policy-path',type=Path,required=True); p.add_argument('--dataset-root',type=Path,required=True); p.add_argument('--repo-id',required=True); p.add_argument('--torque-mode',choices=('none','original','zero','shuffle'),default='none'); p.add_argument('--seed',type=int,default=999); p.add_argument('--device',choices=('cpu','cuda'),default='cpu'); a=p.parse_args()
def send(c,x):
 b=pickle.dumps(x,protocol=5); c.sendall(struct.pack('!Q',len(b))+b)
def recv(c):
 h=b''
 while len(h)<8: h+=c.recv(8-len(h))
 n=struct.unpack('!Q',h)[0]; b=b''
 while len(b)<n: b+=c.recv(min(1<<20,n-len(b)))
 return pickle.loads(b)
def rgb(x): return torch.from_numpy(np.asarray(Image.fromarray(x).resize((224,224),Image.Resampling.BILINEAR))[None]).permute(0,3,1,2).float()/255.
# Draccus compatibility: omit one optional Literal-typed config field while parsing.
raw_cfg=json.loads((a.policy_path/'config.json').read_text()); raw_cfg.pop('tactile_token_mode',None)
compat_dir=Path(tempfile.mkdtemp(prefix='factory_eval_cfg_')); (compat_dir/'config.json').write_text(json.dumps(raw_cfg))
cfg=PreTrainedConfig.from_pretrained(compat_dir); cfg.pretrained_path=str(a.policy_path); cfg.device=a.device
if a.torque_mode=='none': cfg.use_torque_lstm=False
dataset_path=a.dataset_root if (a.dataset_root/'meta').exists() else a.dataset_root/a.repo_id
print('[FACTORY_CLIENT] building_policy',flush=True)
meta=LeRobotDatasetMetadata(a.repo_id,root=dataset_path); policy=make_policy(cfg=cfg,ds_meta=meta); policy.eval(); print('[FACTORY_CLIENT] policy_ready device='+str(next(policy.parameters()).device),flush=True)
pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=str(a.policy_path),preprocessor_overrides={'device_processor':{'device':a.device}}); print('[FACTORY_CLIENT] processors_ready',flush=True)
print('[FACTORY_CLIENT] connecting',flush=True)
c=socket.socket()
for _ in range(120):
 try: c.connect((a.host,a.port)); break
 except OSError: time.sleep(1)
else: raise RuntimeError('server unavailable')
rng=np.random.default_rng(a.seed)
while True:
 m=recv(c); print('[FACTORY_CLIENT] received_'+m['kind'],flush=True)
 if m['kind']=='done': print('[FACTORY_CLIENT] done',flush=True); break
 if m['kind']=='episode': print('[FACTORY_CLIENT]',m,flush=True); continue
 t=m['torque'].astype(np.float32)
 if a.torque_mode=='zero': t=np.zeros_like(t)
 elif a.torque_mode=='shuffle': t=t[rng.permutation(len(t))]
 batch={'observation.state':torch.from_numpy(m['state'][None]),'observation.images.camera1':rgb(m['camera1']),'observation.images.camera2':rgb(m['camera2']),'task':['Insert the peg into the hole']}
 if a.torque_mode!='none': batch['observation.gripper_torque']=torch.from_numpy(t[None])
 print('[FACTORY_CLIENT] inference_begin',flush=True)
 with torch.inference_mode(): u=post(policy.select_action(pre(batch))).detach().float().cpu().numpy()[0]
 print('[FACTORY_CLIENT] inference_done action='+repr(u.tolist()),flush=True)
 send(c,{'kind':'action','action':u})
c.close()
