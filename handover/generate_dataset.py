import random
import sys
sys.path.append("../")

import numpy as np
import trimesh
import os
import yaml
from scipy.spatial.kdtree import KDTree
from utils import scene_util, pc_util
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from multiprocessing import Pool
from tqdm import tqdm

dir_path = os.path.dirname(os.path.realpath(__file__))
with open("./config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)


def grasp_registration(img_id, use_base_coordinate=True, split="train", vis=False, save=False):
    # farthest_points = cfg["dataset"]["farthest_points"]
    # crop_radius = cfg["dataset"]["crop_radius"]
    label_path = cfg["dataset"]["label_path"]

    points = scene_util.load_scene_pc(img_id, use_base_coordinate=use_base_coordinate, split=split)
    points = pc_util.crop_pc(points)
    mesh, obj, transform = scene_util.load_scene(img_id, use_base_coordinate=use_base_coordinate, split=split)

    grasp_name = "obj_" + str(obj["obj_id"]).zfill(6) + ".npy"
    grasp_path = os.path.join(label_path, grasp_name)
    grasp_config = np.load(grasp_path, allow_pickle=True).item()

    trans_grasp_config = {}
    for k in grasp_config:
        trans_grasp_config[k] = {}
        point_pos = grasp_config[k]["point"][:3].reshape((1, 3))
        taxonomies = grasp_config[k]["tax_name"]

        point_pos = trimesh.PointCloud(point_pos)
        point_pos.apply_transform(transform)
        trans_point_pos = point_pos.vertices.reshape(3)

        trans_grasp_config[k]["point"] = trans_point_pos
        trans_grasp_config[k]["tax_name"] = taxonomies
        trans_grasp_config[k]["DLR_init"] = []

        for taxonomy in taxonomies:
            trans_grasp_config[k][taxonomy] = []
            for grasp in grasp_config[k][taxonomy]:
                grasp_pos = grasp[:3].astype(float)
                grasp_quat = grasp[3:7].astype(float)
                joint = grasp[8:].astype(float)

                T_hand = trimesh.transformations.translation_matrix(grasp_pos)
                R_hand = trimesh.transformations.quaternion_matrix(grasp_quat)
                matrix_hand = trimesh.transformations.concatenate_matrices(T_hand, R_hand)

                trans_matrix_hand = np.dot(transform, matrix_hand)
                trans_grasp_pos = trans_matrix_hand[:3, 3].T
                trans_grasp_quat = trimesh.transformations.quaternion_from_matrix(trans_matrix_hand[:3, :3])

                trans_grasp = np.concatenate([trans_grasp_pos, trans_grasp_quat, joint], axis=-1)
                trans_grasp_config[k][taxonomy].append(trans_grasp)

    for k in trans_grasp_config:
        taxonomies = trans_grasp_config[k]["tax_name"]
        for taxonomy in taxonomies:
            grasps = trans_grasp_config[k][taxonomy]
            index = np.random.choice(np.arange(len(grasps)))
            trans_grasp_config[k][taxonomy] = grasps[index]

    # trans_point2index = {}
    # ori_grasp_points = []
    # for k in trans_grasp_config:
    #     grasp_point = trans_grasp_config[k]["point"]
    #     trans_point2index[tuple(grasp_point)] = k
    #     ori_grasp_points.append(grasp_point)
    # ori_grasp_points = np.asarray(ori_grasp_points)
    #
    # if vis:
    #     scene = trimesh.Scene()
    #     scene.add_geometry([mesh,
    #                         trimesh.PointCloud(ori_grasp_points, colors=[0, 255, 0]),
    #                         trimesh.PointCloud(points, colors=[255, 0, 0])])
    #     scene.show()

    taxonomies = grasp_dict_20f.keys()
    scene_grasp = {}
    for taxonomy in taxonomies:
        scene_grasp[taxonomy] = {"0": [], "1": []}

    for k_p in trans_grasp_config:
        kp_grasps = trans_grasp_config[k_p]
        for taxonomy in taxonomies:
            if taxonomy not in kp_grasps["tax_name"]:
                scene_grasp[taxonomy]["0"].append(kp_grasps["point"])
            else:
                label = np.concatenate([kp_grasps["point"], kp_grasps[taxonomy]])
                scene_grasp[taxonomy]["1"].append(label)

    for taxonomy in taxonomies:
        scene_grasp[taxonomy]["0"] = np.asarray(scene_grasp[taxonomy]["0"])
        scene_grasp[taxonomy]["1"] = np.asarray(scene_grasp[taxonomy]["1"])

    point_grasp_dict = {}
    grasp_tree = KDTree(points)

    for taxonomy in taxonomies:
        point_grasp_dict[taxonomy] = {}
        if taxonomy == "DLR_init":
            all_grasp_point = scene_grasp[taxonomy]["0"]
            points_query = grasp_tree.query_ball_point(all_grasp_point, 0.002)
            points_query = [item for sublist in points_query for item in sublist]
            points_query = list(set(points_query))

            for index in points_query:
                point_grasp_dict[taxonomy][index] = -1
        else:
            if len(scene_grasp[taxonomy]["1"]) != 0:
                good_point = scene_grasp[taxonomy]["1"][:, :3]
                points_query = grasp_tree.query_ball_point(good_point, 0.002)
                for i, pq in enumerate(points_query):
                    if pq:
                        for index in pq:
                            point_grasp_dict[taxonomy][index] = scene_grasp[taxonomy]["1"][i]
                            point_grasp_dict[taxonomy][index][:3] = points[index].reshape(3)

    if vis:
        obj_points = trimesh.PointCloud(points, colors=[80, 200, 200])
        for taxonomy in taxonomies:
            for point_index in point_grasp_dict[taxonomy]:
                if isinstance(point_grasp_dict[taxonomy][point_index], int):
                    continue
                grasp = point_grasp_dict[taxonomy][point_index]
                grasp_point = grasp[:3].reshape((1, 3))

                print(grasp_point in points)

                grasp_pos = grasp[3: 6]
                grasp_quat = grasp[6:10]
                joint = grasp[10:]

                grasp_point = trimesh.PointCloud(grasp_point, colors=[255, 0, 0])
                hand_mesh = scene_util.load_hand(grasp_pos, grasp_quat, joint, color=[0, 255, 255])

                scene = trimesh.Scene()
                scene.add_geometry([mesh, hand_mesh, obj_points, grasp_point])
                scene.show()

    if save:
        if split == "train" or split == "val":
            data_out_path = cfg["train"]["data_out_path"]
            label_out_path = cfg["train"]["label_out_path"]
        else:
            data_out_path = cfg["test"]["data_out_path"]
            label_out_path = cfg["test"]["label_out_path"]
        data_save_path = os.path.join(data_out_path, str(img_id // 1000).zfill(6))
        label_save_path = os.path.join(label_out_path, str(img_id // 1000).zfill(6))

        if not os.path.exists(data_save_path):
            os.makedirs(data_save_path)
        if not os.path.exists(label_save_path):
            os.makedirs(label_save_path)

        point_path = os.path.join(data_save_path, str(img_id % 1000).zfill(6) + "_point.npy")
        grasp_path = os.path.join(label_save_path, str(img_id % 1000).zfill(6) + "_label.npy")

        np.save(point_path, points)
        np.save(grasp_path, point_grasp_dict)

        print(f"scene_{str(img_id).zfill(6)} finished!")


def parallel_grasp_registration(proc):
    p = Pool(processes=proc)
    res_list = []
    for i in range(0, cfg["network"]["test"]["num_images"]):
        res_list.append(p.apply_async(grasp_registration, (i, True, "test", False, True,)))
    p.close()
    p.join()
    for res in tqdm(res_list):
        res.get()


if __name__ == "__main__":
    parallel_grasp_registration(10)
