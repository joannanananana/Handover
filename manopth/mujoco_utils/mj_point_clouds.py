import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import trimesh
import os


def quat2mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix")
        raise ValueError

    rot_mat = trimesh.transformations.quaternion_matrix(quat)
    return rot_mat[:3, :3]


def rotMatList2NPRotMat(rot_mat_arr):
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


def posRotMat2Mat(pos, rot_mat):
    t_mat = trimesh.transformations.translation_matrix(pos)
    r_mat = trimesh.transformations.rotation_from_matrix(rot_mat)
    mat = trimesh.transformations.concatenate_matrices(t_mat, r_mat)
    return mat


def cammat2o3d(cam_mat, width, height):
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class PointCloudGenerator:
    def __init__(self, sim, min_bound=None, max_bound=None):
        self.sim = sim
        self.img_width = 1280
        self.img_height = 720
        self.cam_names = self.sim.model.camera_names
        self.target_bounds = None
        if min_bound and max_bound:
            self.target_bounds = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

        self.cam_mats = []
        for cam_id in range(len(self.cam_names)):
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            f = self.img_height / (2 * math.tan(fovy / 2))
            cam_mat = np.array(((f, 0, self.img_width / 2), (0, f, self.img_height / 2), (0, 0, 1)))
            self.cam_mats.append(cam_mat)

    def generateCroppedPointCloud(self, save_img_dir=None):
        o3d_clouds = []
        cam_poses = []
        for cam_i in range(len(self.cam_names)):
            depth_img, rgb = self.captureImage(cam_i)
            if save_img_dir:
                self.saveImg(depth_img, save_img_dir, "depth_test_" + str(cam_i))
                color_img = self.captureImage(cam_i, False)
                self.saveImg(color_img, save_img_dir, "color_test_" + str(cam_i))
            od_cammat = cammat2o3d(self.cam_mats[cam_i], self.img_width, self.img_height)
            od_depth = o3d.geometry.Image(depth_img)
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(od_depth, od_cammat)

            cam_body_id = self.sim.model.cam_bodyid[cam_i]
            cam_pos = self.sim.model.body_pos[cam_body_id]
            c2b_r = rotMatList2NPRotMat(self.sim.model.cam_mat0[cam_i])

            b2w_r = quat2mat([0, 1, 0, 0])
            c2w_r = np.matmul(c2b_r, b2w_r)
            c2w = posRotMat2Mat(cam_pos, c2w_r)
            transformed_cloud = o3d_cloud.transform(c2w)

            if self.target_bounds:
                transformed_cloud = transformed_cloud.crop(self.target_bounds)

            transformed_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))
            transformed_cloud.orient_normals_towards_camera_location(cam_pos)

            o3d_clouds.append(transformed_cloud)

        combined_cloud = o3d.geometry.PointCloud()
        for cloud in o3d_clouds:
            combined_cloud += cloud
        return combined_cloud, rgb

    def depth_img2meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def verticalFlip(self, img):
        return np.flip(img, axis=0)

    def captureImage(self, cam_ind, capture_depth=True):
        rendered_images = self.sim.render(self.img_width, self.img_height, camera_name=self.cam_names[cam_ind], depth=capture_depth)
        if capture_depth:
            img, depth = rendered_images
            depth = self.verticalFlip(depth)
            real_depth = self.depth_img2meters(depth)
            return real_depth, img
        else:
            img = rendered_images
            return self.verticalFlip(img)

    def saveImg(self, img, filepath, filename):
        normalized_image = img / img.max() * 255
        normalized_image = normalized_image.astype(np.uint8)
        im = Image.fromarray(normalized_image)
        im.save(os.path.join(filepath, filename + ".jpg"))
