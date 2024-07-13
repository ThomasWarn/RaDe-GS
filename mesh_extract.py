import os
import torch
from random import randint
import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
import math
import numpy as np
from scene.cameras import Camera
from gaussian_renderer import render
import open3d as o3d
import open3d.core as o3c
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import os
import torch
import numpy as np
import math
from scene import Scene, GaussianModel
from argparse import ArgumentParser, Namespace
from scene.cameras import Camera
from gaussian_renderer import render
import open3d as o3d
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos

def load_camera_colmap(args):
    scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
    return cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)

def extract_mesh(dataset, pipe, checkpoint_iterations=None):
    gaussians = GaussianModel(dataset.sh_degree)
    output_path = os.path.join(dataset.model_path, "point_cloud")
    iteration = 0
    if checkpoint_iterations is None:
        for folder_name in os.listdir(output_path):
            iteration = max(iteration, int(folder_name.split('_')[1]))
    else:
        iteration = checkpoint_iterations
    output_path = os.path.join(output_path, "iteration_" + str(iteration), "point_cloud.ply")

    gaussians.load_ply(output_path)
    print(f'Loaded gaussians from {output_path}')
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_cam_list = load_camera_colmap(dataset)

    depth_list = []
    color_list = []
    alpha_thres = 0.51
    for viewpoint_cam in viewpoint_cam_list:
        # Rendering offscreen from that camera 
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        rendered_img = torch.clamp(render_pkg["render"], min=0, max=1.0).cpu().numpy()
        color_list.append(rendered_img)
        depth = render_pkg["middepth"].clone()
        if viewpoint_cam.gt_mask is not None:
            depth[(viewpoint_cam.gt_mask < 0.5)] = 0
        depth[render_pkg["mask"] < alpha_thres] = 0
        depth_list.append(depth[0].cpu().numpy())
    torch.cuda.empty_cache()
    o3d_device = o3d.core.Device("CPU:0")
    #ScalableTSDFVolume
    voxel_length = 0.01  #voxel size
    sdf_trunc = 0.04     #truncation distance
    tsdf_volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )
    counter_temp = 0
    for color, depth, viewpoint_cam in zip(color_list, depth_list, viewpoint_cam_list):
        counter_temp += 1
        print(counter_temp)
        depth_o3d = o3d.geometry.Image(depth)
        
        color = np.transpose(color, (1, 2, 0))  #transpose to shape (H, W, 3)
        color = np.ascontiguousarray(color * 255).astype(np.uint8)
        color_o3d = o3d.geometry.Image(color)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color = color_o3d, depth = depth_o3d, depth_scale = 1.0, depth_trunc = 8.0, convert_rgb_to_intensity=False
        )

        W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
        fx = W / (2 * math.tan(viewpoint_cam.FoVx / 2.))
        fy = H / (2 * math.tan(viewpoint_cam.FoVy / 2.))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, W/2, H/2)
        extrinsic = viewpoint_cam.extrinsic.cpu().numpy().astype(np.float64)

        tsdf_volume.integrate(
            image = rgbd_image,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
        )#1.0, 8.0
        
    mesh = tsdf_volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(dataset.model_path, "recon.ply"), mesh)
    print("done!")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    with torch.no_grad():
        extract_mesh(lp.extract(args), pp.extract(args), args.checkpoint_iterations)

    
    
