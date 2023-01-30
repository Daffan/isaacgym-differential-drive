from matplotlib import pyplot as plt
import matplotlib
import math
from PIL import Image
import imageio
import os

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from isaacgym.terrain_utils import *

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Jackal Isaac")

# configure sim
sim_params = gymapi.SimParams()
#sim_params.up_axis = gymapi.UP_AXIS_Z
#sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.static_friction = 100
plane_params.dynamic_friction = 50
gym.add_ground(sim, plane_params)

asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')
asset_file = "urdf/jackal/urdf/jackal.urdf"
asset = gym.load_asset(sim, asset_root, asset_file, gymapi.AssetOptions())

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
cylinder_asset = gym.create_capsule(sim, 0.075, 1, asset_options)

spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)

envs = []
actor_handles = []
camera_handles = []
depth_handles = []
for i in range(4):
    env = gym.create_env(sim, lower, upper, 2)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(2.25, 0.0, 0.0)
    pose.r = (
        gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), math.pi * -0.5) *
        gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), math.pi * -0.5)
    )

    asset_options = gymapi.AssetOptions()

    actor_handle = gym.create_actor(env, asset, pose, "Jackal", i, 0)
    actor_handles.append(actor_handle)

    props = gym.get_actor_dof_properties(env, actor_handle)
    props["driveMode"].fill(gymapi.DOF_MODE_VEL)
    props["stiffness"].fill(1.0)
    props["damping"].fill(10000000.0)
    gym.set_actor_dof_properties(env, actor_handle, props)

    num_dofs = gym.get_actor_dof_count(env, actor_handle)
    vel_targets = np.ones(num_dofs).astype('f') * math.pi 
    gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)

    # Code to set up the camera
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1024
    camera_props.height = 1024
    camera_props.horizontal_fov = 90
    camera_handle = gym.create_camera_sensor(env, camera_props)
    gym.set_camera_location(camera_handle, env, gymapi.Vec3(2.,10.,1), gymapi.Vec3(2.,0.,0.9999))
    camera_handles.append(camera_handle)

    depth_sensors = []
    #for angle in [0, 90, 180, 270]:
    for angle in [0, 270, 180, 90]:
        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 256
        camera_props.horizontal_fov = 90
        camera_handle = gym.create_camera_sensor(env, camera_props)
        local_transform = gymapi.Transform()
        #local_transform.p = gymapi.Vec3(-0.12, 0, 0.3)
        local_transform.p = gymapi.Vec3(0, 0, 0.3)
        local_transform.r = (
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(90)) * 
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.radians(90)) *
            gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(angle))
        )
        gym.attach_camera_to_body(camera_handle, env, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
        depth_sensors.append(camera_handle)

    depth_handles.append(depth_sensors)
if not os.path.exists("images"):
    os.mkdir("images")

for k in range(1000):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    if k % 200 == 0:
        # Get the color image of the last env
        color_image = gym.get_camera_image(sim, envs[-1], camera_handles[-1], gymapi.IMAGE_COLOR)
        matplotlib.image.imsave('images/topdown_%d.png' %k, color_image.reshape(1024, 1024, 4))

        depth_images = []
        for d in depth_handles[-1]:
            dd = gym.get_camera_image(sim, envs[-1], d, gymapi.IMAGE_DEPTH).reshape(256, 256)
            depth_images.append(np.clip(-dd, a_min=0, a_max=20))

        matplotlib.image.imsave('images/depth_%d.png' %k, np.concatenate(depth_images, axis=1))
