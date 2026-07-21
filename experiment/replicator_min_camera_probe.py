#!/usr/bin/env python3
import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
print('[REP_MIN] launching_app', flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print('[REP_MIN] app_launched', flush=True)

try:
    import os
    import numpy as np
    import omni.replicator.core as rep
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.api.world import World
    from pxr import UsdGeom

    world = World(physics_dt=0.01, rendering_dt=0.01, backend='torch', device='cpu')
    prim_utils.create_prim('/World/Cube', prim_type='Cube', translation=(0.0, 0.0, 0.5), scale=(0.5, 0.5, 0.5))
    cam_prim = prim_utils.create_prim('/World/Camera', prim_type='Camera', translation=(2.0, 2.0, 2.0), orientation=(0.33985113, 0.17591988, 0.42470818, 0.82047324))
    _ = UsdGeom.Camera(cam_prim)
    rp = rep.create.render_product('/World/Camera', resolution=(84, 84))
    if not isinstance(rp, str):
        rp = rp.path
    print(f'[REP_MIN] render_product={rp}', flush=True)
    ann = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
    ann.attach(rp)
    if os.environ.get('REP_MIN_ADD_HYDRA', '0') == '1':
        import omni.usd
        ctx = omni.usd.get_context()
        try:
            print(f'[REP_MIN] hydra_before={ctx.get_attached_hydra_engine_names()}', flush=True)
        except Exception as exc:
            print(f'[REP_MIN] hydra_before_error={exc!r}', flush=True)
        omni.usd.add_hydra_engine('rtx', ctx)
        try:
            print(f'[REP_MIN] hydra_after={ctx.get_attached_hydra_engine_names()}', flush=True)
        except Exception as exc:
            print(f'[REP_MIN] hydra_after_error={exc!r}', flush=True)
    world.reset()
    for i in range(10):
        world.step(render=True)
        world.render()
        data = ann.get_data()
        arr = np.asarray(data)
        print(f'[REP_MIN] i={i} shape={arr.shape} dtype={arr.dtype} size={arr.size} min={(arr.min() if arr.size else None)} max={(arr.max() if arr.size else None)}', flush=True)
    simulation_app.close()
except BaseException as exc:
    print(f'[REP_MIN] exception type={type(exc).__name__} repr={exc!r}', flush=True)
    traceback.print_exc()
    try:
        simulation_app.close()
    except Exception:
        pass
    raise
