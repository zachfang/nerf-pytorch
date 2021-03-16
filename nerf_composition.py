import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


import matplotlib.pyplot as plt
import run_nerf
import load_blender

torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

basedir = './logs'
static_nerf = 'tri_empty_sink_80cm'
dynamic_nerf = "tri_mug_50cm_sphere"
static_config = os.path.join(basedir, static_nerf, 'config.txt')
dynamic_config = os.path.join(basedir, dynamic_nerf, 'config.txt')
print('Static args:')
#print(open(static_config, 'r').read())
parser = run_nerf.config_parser()
static_args = parser.parse_args('--config {} '.format(static_config))
print(static_args)
print('Dynamic args:')
#print(open(static_config, 'r').read())
#parser = run_nerf.config_parser()
dynamic_args = parser.parse_args('--config {} '.format(dynamic_config))
print(dynamic_args)

# Multi-GPU
args.n_gpus = torch.cuda.device_count()
print(f"Using {args.n_gpus} GPU(s).")

H, W = 480, 848
focal = .5 * W / np.tan(.5 * 1.2037270069122314)
hwf = [H, W, focal]

# Create nerf model
_, static_render_kwargs_test, _, _, _ = run_nerf.create_nerf(static_args)
_, dynamic_render_kwargs_test, _, _, _ = run_nerf.create_nerf(dynamic_args)

bds_dict = {
    'near' : 0.2,
    'far' : 1.5,
}
static_render_kwargs_test.update(bds_dict)
dynamic_render_kwargs_test.update(bds_dict)

"""
with torch.no_grad():
    c2w = load_blender.pose_spherical(0.0, -90.0, 0.8)

    rgb, disp, _, _ = run_nerf.render(H, W, focal, c2w=c2w[:3, :4], **render_kwargs_test)
    #plt.imshow(rgb.cpu())
    #plt.show()
    print('Done rendering')

    N = 256
    x = np.linspace(-0.3, 0.3, N+1)
    y = np.linspace(-0.3, 0.3, N+1)
    z = np.linspace(-0.1, 0.5, N+1)

    query_pts = np.stack(np.meshgrid(x, y, z), -1).astype(np.float32)
    print(query_pts.shape)
    sh = query_pts.shape
    flat = torch.from_numpy(query_pts.reshape([-1,3]))

    net_fn = render_kwargs_test["network_query_fn"]
    fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:].to(device), viewdirs=torch.zeros_like(flat[i0:i1]).to(device), network_fn=render_kwargs_test['network_fine'])
    chunk = 1024*64
    raw = np.concatenate([fn(i, i+chunk).cpu().numpy() for i in range(0, flat.shape[0], chunk)], 0)
    raw = np.reshape(raw, list(sh[:-1]) + [-1])
    sigma = np.maximum(raw[...,-1], 0.)

    print(raw.shape)
    plt.hist(np.maximum(0,sigma.ravel()), log=True)
    plt.show()
"""
"""
import mcubes

threshold = 10.
print('fraction occupied', np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape)
import trimesh

mesh = trimesh.Trimesh(vertices / N - .5, triangles)
#mesh.export("~/sink.ply")
mesh.show()


#os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender

scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))

# Set up the camera -- z-axis away from the scene, x-axis right, y-axis up
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

camera_pose = load_blender.pose_spherical(-20., -40., 1.).cpu().numpy()
nc = pyrender.Node(camera=camera, matrix=camera_pose)
scene.add_node(nc)

# Set up the light -- a point light in the same spot as the camera
light = pyrender.PointLight(color=np.ones(3), intensity=4.0)
nl = pyrender.Node(light=light, matrix=camera_pose)
scene.add_node(nl)

# Render the scene
r = pyrender.OffscreenRenderer(640, 480)
color, depth = r.render(scene)

plt.imshow(color)
plt.show()
plt.imshow(depth)
plt.show()
"""