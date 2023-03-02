import numpy as np
import trimesh
import os
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from utils import scene_util, common_util
import copy
import yaml
from torch.utils.data import Dataset, DataLoader
import torch


with open("config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


class GraspDataset(Dataset):
    def __init__(self, split="train", vis=False, vis_all_grasp=False, vis_by_taxonomy=False):
        self.num_imgs = cfg["network"]["train"]["num_images"]
        self.taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]
        self.num_points = cfg["network"]["train"]["num_points"]
        self.R_hand = np.load("utils/R_hand.npy")
        self.R_hand_inv = common_util.inverse_transform_matrix(self.R_hand[:3, :3], self.R_hand[:3, 3])

        if split == "train" or split == "val":
            self.data_path = cfg["train"]["data_out_path"]
            self.label_path = cfg["train"]["label_out_path"]
        else:
            self.data_path = cfg[split]["data_out_path"]
            self.label_path = cfg[split]["label_out_path"]

        self.vis = vis
        self.vis_all_grasp = vis_all_grasp
        self.vis_by_taxonomy = vis_by_taxonomy
        self.split = split
        if split == "train":
            self.imgIds = list(range(0, 16000))
        elif split == "val":
            self.imgIds = list(range(16000, 20000))
        else:
            self.imgIds = list(range(cfg["network"]["test"]["num_images"]))

    def __getitem__(self, index):
        img_id = self.imgIds[index]
        img_folder = str(img_id // 1000).zfill(6)
        img_name = str(img_id % 1000).zfill(6)
        data_file = os.path.join(self.data_path, img_folder, img_name + "_point.npy")
        label_file = os.path.join(self.label_path, img_folder, img_name + "_label.npy")

        points = np.load(data_file)
        grasps = np.load(label_file, allow_pickle=True).item()

        point_data = dict()
        point_data["point"] = points

        center = np.mean(points, axis=0)
        point_data["norm_point"] = points - center

        shape = points.shape
        axis_min_values = np.min(points, axis=0)
        axis_max_values = np.max(points, axis=0)
        x_sigma, y_sigma, z_sigma = (axis_max_values.squeeze() - axis_min_values.squeeze()) / 20
        x_noise = np.random.normal(0, x_sigma, shape[0])
        y_noise = np.random.normal(0, y_sigma, shape[0])
        z_noise = np.random.normal(0, z_sigma, shape[0])
        noise = np.dstack((x_noise, y_noise, z_noise)).squeeze()
        noise_point = points + noise

        grasp_point_label = np.zeros(len(points))

        for taxonomy in self.taxonomies:
            point_label = copy.deepcopy(grasp_point_label)
            good_points_index = list(grasps[taxonomy].keys())
            point_label[good_points_index] = 1

            hand_conf = np.zeros((len(points), 25))
            if grasps[taxonomy]:
                hand_conf_label = np.concatenate(list(grasps[taxonomy].values())).reshape(-1, 30)[:, 3:]
                pos = hand_conf_label[:, :3]
                quat = hand_conf_label[:, 3:7]
                mat = trimesh.transformations.quaternion_matrix(quat)
                if len(mat.shape) < 3:
                    mat = mat[np.newaxis, :, :]
                mat[:, :3, 3] = pos

                new_mat = np.dot(mat, self.R_hand_inv)
                approach = new_mat[:, :3, 2]

                offset = pos - points[good_points_index]
                depth = np.sum(offset * approach, axis=-1)

                hand_conf[good_points_index, 0] = depth
                hand_conf[good_points_index, 1:5] = quat
                hand_conf[good_points_index, 5:] = hand_conf_label[:, 7:]

                joint_init = np.asarray(grasp_dict_20f[taxonomy]["joint_init"]) * np.pi / 180.0
                joint_final = np.asarray(grasp_dict_20f[taxonomy]["joint_final"]) * np.pi / 180.0
                hand_conf[good_points_index, 5:] = np.clip(hand_conf[good_points_index, 5:], joint_init, joint_final)
                hand_conf[good_points_index, 5:] = (hand_conf[good_points_index, 5:] - joint_init) / \
                                                   (joint_final - joint_init + 0.00001)

            label = np.concatenate((point_label[:, np.newaxis], hand_conf), axis=-1)

            point_data[taxonomy] = label
        if self.num_points <= len(points):
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        for k in point_data:
            point_data[k] = point_data[k][choice]
        noise_point = noise_point[choice]

        if self.vis:
            points = point_data["point"]
            obj_points = trimesh.PointCloud(points, colors=[25, 200, 100])
            obj_points_noise = trimesh.PointCloud(noise_point, colors=[25, 200, 100])
            for taxonomy in self.taxonomies:
                for i in range(len(point_data[taxonomy])):
                    grasp = point_data[taxonomy][i]
                    grasp_point = point_data["point"][i]
                    if grasp[0] == 0:
                        # print("No suitable grasp pose!")
                        continue
                    else:
                        depth = grasp[1]
                        quat = grasp[2:6]
                        R = trimesh.transformations.quaternion_matrix(quat)
                        new_R = np.dot(R, self.R_hand_inv)
                        new_quat = trimesh.transformations.quaternion_from_matrix(new_R)
                        new_quat_mat = trimesh.transformations.quaternion_matrix(new_quat)
                        approach = np.dot(R, self.R_hand_inv)[:3, 2]
                        pos = approach * (depth + 0.1) + grasp_point

                        pos = trimesh.transformations.translation_matrix(pos)
                        mat = trimesh.transformations.concatenate_matrices(pos, new_quat_mat)
                        pos = np.dot(mat, self.R_hand)[:3, 3].reshape(3)

                        joint = grasp[6:]
                        joint_init = np.asarray(grasp_dict_20f[taxonomy]["joint_init"]) * np.pi / 180.0
                        joint_final = np.asarray(grasp_dict_20f[taxonomy]["joint_final"]) * np.pi / 180.0
                        joint = joint * (joint_final - joint_init + 0.00001) + joint_init

                        hand_mesh = scene_util.load_hand(pos, quat, joint, color=[0, 200, 200])
                        approach_ray = trimesh.load_path(np.hstack((
                            grasp_point,
                            grasp_point + approach / 10)).reshape(-1, 3))

                        scene = trimesh.Scene()
                        scene.add_geometry([obj_points,
                                            hand_mesh,
                                            trimesh.PointCloud(grasp_point.reshape((-1, 3)), colors=[255, 0, 0])])

                        scene.show()

                        scene = trimesh.Scene()
                        scene.add_geometry([obj_points_noise,
                                            hand_mesh,
                                            trimesh.PointCloud(grasp_point.reshape((-1, 3)), colors=[255, 0, 0])])

                        scene.show()

        if self.vis_all_grasp:
            points = point_data["point"]
            obj_points = trimesh.PointCloud(points, colors=[255, 0, 0])
            mesh, _, _ = scene_util.load_scene(img_id, use_base_coordinate=True, split=self.split)

            obj_points_noise = trimesh.PointCloud(noise_point, colors=[255, 0, 0])

            scene1 = trimesh.Scene()
            scene1.add_geometry([mesh, obj_points])

            scene2 = trimesh.Scene()
            scene2.add_geometry([mesh, obj_points_noise])

            # scene.show()

            point_index = list(range(len(points)))
            np.random.shuffle(point_index)
            num_grasp = 0

            for i in point_index:
                if num_grasp >= 1:
                    break
                curr_point = points[i]
                feasible_point = []
                for t in range(len(self.taxonomies)):
                    tax = self.taxonomies[t]
                    if int(point_data[tax][i, 0]) == 1:
                        feasible_point.append(t)
                if not feasible_point:
                    curr_pc = trimesh.PointCloud(curr_point.reshape(-1, 3), colors=[0, 0, 0])
                    scene1.add_geometry(curr_pc)
                    scene2.add_geometry(curr_pc)
                    continue
                else:
                    selected_tax_index = np.random.choice(feasible_point)
                    selected_tax = self.taxonomies[selected_tax_index]
                    grasp = point_data[selected_tax][i]

                    depth = grasp[1]
                    quat = grasp[2:6]
                    R = trimesh.transformations.quaternion_matrix(quat)
                    new_R = np.dot(R, self.R_hand_inv)
                    new_quat = trimesh.transformations.quaternion_from_matrix(new_R)
                    new_quat_mat = trimesh.transformations.quaternion_matrix(new_quat)
                    approach = np.dot(R, self.R_hand_inv)[:3, 2]
                    pos = approach * (depth + 0.1) + curr_point

                    pos = trimesh.transformations.translation_matrix(pos)
                    mat = trimesh.transformations.concatenate_matrices(pos, new_quat_mat)
                    pos = np.dot(mat, self.R_hand)[:3, 3].reshape(3)

                    joint = grasp[6:]
                    joint_init = np.asarray(grasp_dict_20f[selected_tax]["joint_init"]) * np.pi / 180.0
                    joint_final = np.asarray(grasp_dict_20f[selected_tax]["joint_final"]) * np.pi / 180.0
                    joint = joint * (joint_final - joint_init + 0.00001) + joint_init

                    hand_mesh = scene_util.load_hand(pos, quat, joint, color=cfg["color"][selected_tax])
                    # approach_ray = trimesh.load_path(np.hstack((
                    #     curr_point,
                    #     curr_point + approach / 10)).reshape(-1, 3))

                    scene1.add_geometry([obj_points, hand_mesh])
                    scene2.add_geometry([obj_points_noise, hand_mesh])
                    num_grasp += 1

            scene1.show()
            scene2.show()

        if self.vis_by_taxonomy:
            points = point_data["point"]
            obj_points = trimesh.PointCloud(points, colors=[255, 0, 0])

            mesh, _, _ = scene_util.load_scene(img_id, use_base_coordinate=True, split=self.split)

            scene = trimesh.Scene()
            # scene.add_geometry([mesh, obj_points])
            scene.add_geometry(mesh)

            scene.show()

            point_index = list(range(len(points)))
            np.random.shuffle(point_index)

            for tax in self.taxonomies:
                obj_points = trimesh.PointCloud(points, colors=[255, 0, 0])
                scene = trimesh.Scene()
                # scene.add_geometry([mesh, obj_points])
                scene.add_geometry(mesh)
                num_grasp = 0
                for i in point_index:
                    if int(point_data[tax][i, 0]) == 1:
                        curr_point = points[i]
                        curr_pc = trimesh.PointCloud(curr_point.reshape(-1, 3), colors=[0, 255, 0])
                        # scene.add_geometry(curr_pc)
                        if num_grasp >= 5:
                            continue
                        grasp = point_data[tax][i]
                        depth = grasp[1]
                        quat = grasp[2:6]
                        R = trimesh.transformations.quaternion_matrix(quat)
                        new_R = np.dot(R, self.R_hand_inv)
                        new_quat = trimesh.transformations.quaternion_from_matrix(new_R)
                        new_quat_mat = trimesh.transformations.quaternion_matrix(new_quat)
                        approach = np.dot(R, self.R_hand_inv)[:3, 2]
                        pos = approach * (depth + 0.1) + curr_point

                        pos = trimesh.transformations.translation_matrix(pos)
                        mat = trimesh.transformations.concatenate_matrices(pos, new_quat_mat)
                        pos = np.dot(mat, self.R_hand)[:3, 3].reshape(3)

                        joint = grasp[6:]
                        joint_init = np.asarray(grasp_dict_20f[tax]["joint_init"]) * np.pi / 180.0
                        joint_final = np.asarray(grasp_dict_20f[tax]["joint_final"]) * np.pi / 180.0
                        joint = joint * (joint_final - joint_init + 0.00001) + joint_init

                        hand_mesh = scene_util.load_hand(pos, quat, joint, color=cfg["color"][tax])

                        scene.add_geometry(hand_mesh)
                        num_grasp += 1
                    else:
                        continue
                scene.show()

        return point_data, img_id

    def __len__(self):
        return len(self.imgIds)


if __name__ == "__main__":
    dataset = GraspDataset(vis=False, split="train", vis_all_grasp=False, vis_by_taxonomy=False)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

    for i, (data, index) in enumerate(dataloader):
        # print("current index:", index.item())
        # if i in [32, 33, 34, 35, 36, 37, 38]:
        #     print("current index:", index.item())
        # else:
        #     continue
        print(data.keys())
        print(data["Parallel_Extension"])
        exit()
        # print(data["point"])
        print(torch.mean(data["point"], dim=1))
        # print(data["point"] - data["norm_point"])
        # print(data["point"] - torch.mean(data["point"], dim=1) - data["norm_point"])
        # exit()
        shape = data["point"].numpy().shape
        noise = []
        axis_min_values = np.min(data["point"].numpy(), axis=1)
        axis_max_values = np.max(data["point"].numpy(), axis=1)
        sigma = (axis_max_values - axis_min_values) / 20
        print(sigma[0])
        exit()
        for i in range(sigma.shape[0]):
            x_sigma, y_sigma, z_sigma = sigma[i]
            x_noise = np.random.normal(0, x_sigma, shape[1])
            y_noise = np.random.normal(0, y_sigma, shape[1])
            z_noise = np.random.normal(0, z_sigma, shape[1])
            bat_noise = np.dstack((x_noise, y_noise, z_noise)).squeeze()
            noise.append(bat_noise)
        noise = np.asarray(noise)
        print(torch.div(torch.Tensor(noise), data["point"]))
        print(data["point"] + np.random.normal(0, .05, data["point"].numpy().shape))
        print(data["norm_point"])
        exit()
        # print(index.item())
        # print(index.shape)
        # print(type(index))
        #     print(torch.any(torch.isnan(data[k])).item())
        # print(index[0])
        # print(data["point"].shape)
        # print(copy.deepcopy(data["point"])[0].shape)
        # exit()
