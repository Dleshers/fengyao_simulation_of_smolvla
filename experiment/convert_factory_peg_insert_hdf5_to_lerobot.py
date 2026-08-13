#!/usr/bin/env python3
"""Convert strict Factory peg insertion HDF5 into a local LeRobot v3 dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def image(x):
    x=np.asarray(x)
    if x.shape[-1]==4: x=x[...,:3]
    if x.shape[:2] != (224,224): x=np.asarray(Image.fromarray(x).resize((224,224),Image.Resampling.BILINEAR))
    return x.astype(np.uint8)
def history(x,t,n=30):
    y=x[max(0,t-n+1):t+1]
    if len(y)<n: y=np.concatenate([np.repeat(y[:1],n-len(y),0),y])
    return y.astype(np.float32)
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--repo-id',required=True)
p.add_argument('--torque-control',choices=('original','zero','shuffle_episode','shuffle_causal'),default='original'); p.add_argument('--seed',type=int,default=1000); p.add_argument('--fps',type=int,default=15); p.add_argument('--use-videos',action='store_true')
p.add_argument('--torque-dim',choices=(1,7),type=int,default=1,help='1 keeps the historical torque norm; 7 preserves signed joint-torque directions.')
p.add_argument('--recovery-repeat',type=int,default=1,help='Repeat each conditional-recovery episode this many times when materializing the dataset.'); p.add_argument('--policy-label-only',action='store_true',help='For contact-recovery raw HDF5, retain history but add only post-contact recovery labels.')
a=p.parse_args(); root=a.output_dir/a.repo_id
if root.exists(): raise FileExistsError(root)
if a.recovery_repeat < 1: p.error('--recovery-repeat must be >=1')
features={'observation.state':{'dtype':'float32','shape':(12,),'names':None},'action':{'dtype':'float32','shape':(6,),'names':None},'observation.images.camera1':{'dtype':'image','shape':(3,224,224),'names':['channels','height','width']},'observation.images.camera2':{'dtype':'image','shape':(3,224,224),'names':['channels','height','width']},'observation.gripper_torque':{'dtype':'float32','shape':(30,a.torque_dim),'names':None}}
ds=LeRobotDataset.create(repo_id=a.repo_id,root=root,fps=a.fps,features=features,robot_type='isaaclab_factory_franka',use_videos=a.use_videos)
with h5py.File(a.input,'r') as f:
    if f.attrs.get('format','') not in ('factory_peg_insert_formal_v1','factory_peg_insert_causal_recovery_v2','factory_peg_insert_conditional_recovery_v3','factory_peg_insert_visual_oneway_v2','factory_peg_insert_contact_recovery_native_v1','factory_peg_insert_contact_recovery_native_v1_balanced64'): raise ValueError('unexpected raw format')
    demos=sorted(f['demos']); print(f'[FACTORY_CONVERT] demos={len(demos)} control={a.torque_control}',flush=True)
    if not demos: raise ValueError('no demos')
    for i,name in enumerate(demos):
        g=f['demos'][name]
        if not bool(g.attrs.get('strict_success',False)): raise ValueError(f'{name} lacks strict success')
        keys=('state','action','rgb_table','rgb_side','joint_torque','applied_wrench')
        if any(k not in g for k in keys): raise KeyError(name)
        state=np.asarray(g['state'],np.float32); act=np.asarray(g['action'],np.float32); jt=np.asarray(g['joint_torque'],np.float32)
        if state.ndim!=2 or state.shape[1]!=12 or act.shape!=(len(state),6) or jt.shape!=(len(state),7): raise ValueError(f'bad shape {name}: {state.shape} {act.shape} {jt.shape}')
        if not(np.isfinite(state).all() and np.isfinite(act).all() and np.isfinite(jt).all()): raise ValueError(f'nonfinite {name}')
        torque=(np.linalg.norm(jt,axis=1,keepdims=True) if a.torque_dim==1 else jt.copy()).astype(np.float32)
        if a.torque_control=='zero': torque[:]=0
        elif a.torque_control=='shuffle_episode': torque=torque[np.random.default_rng(a.seed+i).permutation(len(torque))]
        repeats=a.recovery_repeat if bool(g.attrs.get('recovery_episode',False)) else 1
        print(f'[FACTORY_CONVERT] {i+1}/{len(demos)} {name} frames={len(state)} repeats={repeats}',flush=True)
        for _ in range(repeats):
            frames = 0
            for t in range(len(state)):
                torque_window=history(torque,t)
                if a.torque_control=="shuffle_causal":
                    prefix=torque[:t+1]
                    rng=np.random.default_rng(a.seed + i * 1000003 + t)
                    if len(prefix) >= 30:
                        torque_window=prefix[rng.choice(len(prefix), size=30, replace=False)]
                    else:
                        torque_window=prefix[rng.choice(len(prefix), size=30, replace=True)]
                if a.policy_label_only and "is_policy_label" in g and not bool(g["is_policy_label"][t, 0]):
                    continue
                ds.add_frame({"task":"Insert the peg into the hole","observation.state":state[t],"action":act[t],"observation.images.camera1":image(g["rgb_table"][t]),"observation.images.camera2":image(g["rgb_side"][t]),"observation.gripper_torque":torque_window})
                frames += 1
            if frames:
                ds.save_episode()
ds.finalize(); print(f'[FACTORY_CONVERT] complete root={root}',flush=True)
