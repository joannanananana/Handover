import mujoco_py
from mujoco_py import MjSim, MjViewer, MjRenderContextOffscreen
import math
import numpy as np
from PIL import Image
import trimesh


def generatePointCloud(sim, camera_viewer, vis=False):
    img_width = 640
    img_height = 640

    fovy = sim.model.cam_fovy[0]
    f = 0.5 * img_height / math.tan(fovy * math.pi / 360)
    cx = img_width / 2
    cy = img_height / 2

    depth_img, mask = captureImage(img_width, img_height, sim, camera_viewer)

    xmap, ymap = np.arange(img_height), np.arange(img_width)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth_img
    points_x = (xmap - cx) / f * points_z
    points_y = (ymap - cy) / f * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)

    indicator = np.where(mask == 255)
    object_points = points[indicator].reshape((-1, 3))

    if vis:
        trimesh.PointCloud(object_points).show()

    return object_points


def captureImage(img_width, img_height, sim, camera_viewer, vis=False):
    camera_viewer.render(img_width, img_height, 0)
    rgb, depth = camera_viewer.read_pixels(img_width, img_height, depth=True)

    flipped_rgb = np.flip(rgb, axis=0)

    flag = flipped_rgb[:, :, 0] <= 10
    mask = np.asarray(255 * flag).astype(np.uint8)

    if vis:
        Image.fromarray(flipped_rgb).show()
        Image.fromarray(mask).show()

    flipped_depth = np.flip(depth, axis=0)
    real_depth = depthimg2Meters(flipped_depth, sim)

    return real_depth, mask


def depthimg2Meters(depth, sim):
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


if __name__ == '__main__':
    model = mujoco_py.load_model_from_path("mujoco_exp/objects_xml/002_vhacd_temp.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    model.opt.gravity[-1] = 0

    camera_viewer = MjRenderContextOffscreen(sim, 0)
    generatePointCloud(sim, camera_viewer)

