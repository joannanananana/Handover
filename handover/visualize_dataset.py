import numpy as np
import yaml
import trimesh
import os
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import scene_util, common_util

with open("./config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


def visualization(img_id, split="train"):
    if split == "train" or split == "val":
        data_path = cfg["train"]["data_out_path"]
        label_path = cfg["train"]["label_out_path"]
    else:
        data_path = cfg[split]["data_out_path"]
        label_path = cfg[split]["label_out_path"]

    img_name = str(img_id % 1000).zfill(6)
    img_folder = str(img_id // 1000).zfill(6)

    data_file = os.path.join(data_path, img_folder, img_name + "_point.npy")
    label_file = os.path.join(label_path, img_folder, img_name + "_label.npy")

    points = np.load(data_file)
    grasps = np.load(label_file, allow_pickle=True).item()

    offset = -np.mean(points, axis=0)
    obj_points = trimesh.PointCloud(points, colors=[50, 200, 50])

    T = trimesh.transformations.translation_matrix(offset)
    R_hand_ori = np.load("utils/R_hand.npy")
    R_hand_inv = common_util.inverse_transform_matrix(R_hand_ori[:3, :3], R_hand_ori[:3, 3])

    obj_points.apply_transform(T)

    taxonomies = grasp_dict_20f.keys()

    for taxonomy in taxonomies:
        for point_index in grasps[taxonomy]:
            if isinstance(grasps[taxonomy][point_index], int):
                continue
            grasp = grasps[taxonomy][point_index]
            grasp_point = grasp[:3].reshape((1, 3))
            grasp_pos = grasp[3: 6]
            grasp_quat = grasp[6: 10]
            joint = grasp[10:]

            R_hand = trimesh.transformations.quaternion_matrix(grasp_quat)
            new_matrix_hand = np.dot(R_hand, R_hand_inv)
            new_pos = new_matrix_hand[:3, 3]
            new_R = new_matrix_hand[:3, :3]
            new_quat = trimesh.transformations.quaternion_from_matrix(new_R)
            new_approach = new_matrix_hand[:3, 2]

            new_approach_ray = trimesh.load_path(np.hstack((
                grasp_point,
                grasp_point + new_approach / 10)).reshape(-1, 3))
            common_util.change_ray_color(new_approach_ray, [0, 0, 255])

            grasp_point = trimesh.PointCloud(grasp_point, colors=[255, 0, 0])
            hand_mesh = scene_util.load_hand(grasp_pos, grasp_quat, joint, color=[0, 50, 200])
            new_hand_mesh = scene_util.load_hand(new_pos, new_quat, joint, color=[0, 200, 50])

            grasp_point.apply_transform(T)
            hand_mesh.apply_transform(T)
            new_hand_mesh.apply_transform(T)
            new_approach_ray.apply_transform(T)

            center_of_hand = trimesh.PointCloud(grasp_pos.reshape((-1, 3)), colors=[255, 0, 0])
            center_of_hand.apply_transform(T)

            scene = trimesh.Scene()
            scene.add_geometry([obj_points, grasp_point, hand_mesh, new_approach_ray, center_of_hand])
            scene.show()


if __name__ == "__main__":
    visualization(0)
