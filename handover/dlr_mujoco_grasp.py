import mujoco_py
import os
import numpy as np
import time
from mujoco_utils.mjcf_xml import MujocoXML
from mujoco_utils.mj_point_clouds import PointCloudGenerator
from mujoco_py import load_model_from_xml, MjSim, functions
from scipy.spatial.transform import Rotation
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import GraspDataset
from model import BackbonePointNet2
from matplotlib import pyplot as plt
import math
from utils import common_util, scene_util, hand_util
import random
import json
import trimesh
import glfw
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import yaml
with open("config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

OKGREEN = "\033[92m"
ENDC = "\033[0m"


class MujocoEnv:
    def __init__(self, split="train", vis=False):
        self.xml_path = "assets/new_hand_lifting.xml"
        self.scene_xml_path = "assets/scene.xml"
        mjb_bytestring = mujoco_py.load_model_from_path(self.xml_path).get_mjb()
        self.model = mujoco_py.load_model_from_mjb(mjb_bytestring)
        self.sim = mujoco_py.MjSim(self.model)
        self.vis = vis
        # self.viewer = mujoco_py.MjViewer(self.sim)

        if split == "train" or split == "val":
            self.root_path = "train_dataset/output/bop_data/lm/train_pbr"
        else:
            self.root_path = "test_dataset/output/bop_data/lm/train_pbr"

    def step(self, k=25):
        for _ in range(k):
            self.sim.step()
            if self.vis:
                self.viewer.render()

    def create_scene_obj(self, obj_root_path, index):
        hand_xml = MujocoXML(self.xml_path)

        # camera2base
        file_path = os.path.join(self.root_path, str(index // 1000).zfill(6))

        with open(os.path.join(file_path, "scene_gt.json")) as f:
            gt_objs = json.load(f)[str(index % 1000)]
        with open(os.path.join(file_path, "scene_camera.json")) as f:
            camera_config = json.load(f)[str(index % 1000)]
            # print(camera_config.keys())

        R_w2c = np.asarray(camera_config["cam_R_w2c"]).reshape(3, 3)
        t_w2c = np.asarray(camera_config["cam_t_w2c"]) * 0.001
        c_w = common_util.inverse_transform_matrix(R_w2c, t_w2c)

        for obj in gt_objs:
            obj_id = obj["obj_id"]
            obj_path = os.path.join(obj_root_path, "obj_" + str(obj_id).zfill(6) + "_vhacd.xml")
            obj_xml = MujocoXML(obj_path)

            T_obj = trimesh.transformations.translation_matrix(np.asarray(obj["cam_t_m2c"]) * 0.001)
            quat_obj = trimesh.transformations.quaternion_from_matrix(np.asarray(obj["cam_R_m2c"]).reshape(3, 3))
            R_obj = trimesh.transformations.quaternion_matrix(quat_obj)
            matrix_obj = trimesh.transformations.concatenate_matrices(T_obj, R_obj)
            transform = np.dot(c_w, matrix_obj)

            c2b_T = transform[0:3, 3]
            c2b_R = transform[:3, :3]
            c2b_R = Rotation.from_matrix(c2b_R).as_euler("xyz") * 180 / math.pi
            obj_xml.translate(c2b_T)
            obj_xml.rotate(c2b_R)

            hand_xml.merge(obj_xml, merge_body=True)
        hand_xml.save_model(self.scene_xml_path)
        xml = hand_xml.get_xml()
        return xml

    def update_scene_model(self, xml):
        self.model = mujoco_py.load_model_from_xml(xml)
        self.sim = mujoco_py.MjSim(self.model)
        if self.vis:
            self.viewer = mujoco_py.MjViewer(self.sim)

    def set_hand_pos(self, joint, pos, quat):
        rad = trimesh.transformations.euler_from_quaternion(quat, axes="rxyz")
        joints_angle = np.array([joint[0], joint[1], joint[2],
                                 joint[4], joint[5], joint[6],
                                 joint[8], joint[9], joint[10],
                                 joint[12], joint[13], joint[14],
                                 joint[16], joint[17], joint[18]])

        state = self.sim.get_state()
        state.qpos[0: 3] = pos
        state.qpos[3] = rad[0]
        state.qpos[4] = rad[1]
        state.qpos[5] = rad[2]
        state.qpos[6: 26] = joint

        self.sim.set_state(state)

        self.sim.data.ctrl[0] = pos[0]
        self.sim.data.ctrl[1] = pos[1]
        self.sim.data.ctrl[2] = pos[2]
        self.sim.data.ctrl[3] = rad[0]
        self.sim.data.ctrl[4] = rad[1]
        self.sim.data.ctrl[5] = rad[2]
        self.sim.data.ctrl[6:] = joints_angle

        self.sim.forward()

    def grasp(self, joint):
        if len(joint) == 20:
            joints_angle = np.array([joint[0], joint[1], joint[2],
                                     joint[4], joint[5], joint[6],
                                     joint[8], joint[9], joint[10],
                                     joint[12], joint[13], joint[14],
                                     joint[16], joint[17], joint[18]])
        if len(joint) == 15:
            joints_angle = np.array(joint)
        self.sim.data.ctrl[6:] = joints_angle

    def lift(self):
        self.sim.data.ctrl[7: 9] += 0.2
        self.sim.data.ctrl[10: 12] += 0.2
        self.sim.data.ctrl[13: 15] += 0.2
        self.sim.data.ctrl[16: 18] += 0.2
        self.sim.data.ctrl[19: 21] += 0.2
        # self.sim.data.ctrl[2] += 0.5
        self.open_gravity()

    def get_pointcloud(self):
        pc_gen = PointCloudGenerator(self.sim, min_bound=(-1., -1., -1.), max_bound=(1., 1., 1.))
        cloud_with_normals, rgb = pc_gen.generateCroppedPointCloud()
        return cloud_with_normals, rgb

    def get_depth(self):
        rgb, depth = self.sim.render(1280, 720, camera_name="BHAM.fixed", depth=True)
        plt.imshow(rgb[:, :, :3])
        plt.axis("off")
        plt.show()
        return depth

    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def get_body_mass(self):
        print(self.model.body_mass)

    def get_geom_id(self, name):
        geom_id = self.model.geom_name2id(name)
        return geom_id

    def ray_mesh(self):
        d = functions.mj_rayMesh(self.model, self.sim.data, 2, np.asarray([0, 0, 0]).astype(np.float64),
                                 np.asarray([0, 0, -1]).astype(np.float64))
        return d

    def contact_check(self):
        print("number of contacts", self.sim.data.ncon)
        contact_id = []
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            print("contact", i)
            print("dist", contact.dist)
            print("geom1", contact.geom1, self.sim.model.geom_id2name(contact.geom1))
            print("geom2", contact.geom2, self.sim.model.geom_id2name(contact.geom2))
            geom2_body = self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2]
            print("Contact force on geom2 body", self.sim.data.cfrc_ext[geom2_body])
            print("norm", np.sqrt(np.sum(np.square(self.sim.data.cfrc_ext[geom2_body]))))
            c_array = np.zeros(6, dtype=np.float64)
            print("c_array", c_array)
            functions.mj_contactForce(self.sim.model, self.sim.data, i, c_array)
            print("c_array", c_array)
            contact_id.append(contact.geom2)

        if len(contact_id) == 0:
            return True

    def lift_obj(self):
        self.sim.data.ctrl[2] += 0.2

    def disable_gravity(self):
        self.model.opt.gravity[-1] = 0

    def open_gravity(self):
        self.model.opt.gravity[-1] = -9.8

    def show_model_info(self):
        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, self.model.body_id2name(i)))

        print("\nNumber of geoms: {}".format(self.model.ngeom))
        for i in range(self.model.ngeom):
            print("Geom ID: {}, Geom Name: {}".format(i, self.model.geom_id2name(i)))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print("Joint ID: {}, Joint Name: {}, Limits: {}".format(i, self.model.joint_id2name(i),
                                                                    self.model.jnt_range[i]))

        print("\nNumber of Actuators: {}".format(len(self.sim.data.ctrl)))
        for i in range(len(self.sim.data.ctrl)):
            print("Actuator ID: {}, Controlled Joint: {}, Control Range: {}".format(i, self.model.actuator_id2name(i),
                                                                                    self.model.actuator_ctrlrange[i]))

        print("\n Camera Info: \n")
        for i in range(self.model.ncam):
            print("Camera ID: {}, Camera Name: {}, Camera FOV (y, degrees): {}, "
                  "Position: {}, Orientation: {}".format(i, self.model.camera_id2name(i), self.model.cam_fovy[i],
                                                         self.model.cam_pos0[i], self.model.cam_mat0[i]))

    def depth_to_pointcloud(self, depth, intrinsic_mat, rgb=None):
        fx, fy = intrinsic_mat[0, 0], intrinsic_mat[1, 1]
        cx, cy = intrinsic_mat[0, 2], intrinsic_mat[1, 2]

        xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth / 1000.0
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        mask = points_z > 0
        points_x = points_x[mask]
        points_y = points_y[mask]
        points_z = points_z[mask]
        points = np.stack([points_x, points_y, points_z], axis=-1)

        if rgb:
            points_rgb = rgb[mask]
        else:
            return points, None
        return points, points_rgb

    def add_one_obj(self, obj_root_path, obj_t, obj_r, random_pos=True):
        hand_xml = MujocoXML(self.xml_path)
        obj_xml = MujocoXML(obj_root_path)
        obj_xml.translate(np.asarray(self.sim.data.get_body_xpos("BHAM.floor")))
        if random_pos:
            obj_xml.translate(obj_t)
            obj_xml.rotate(obj_r)
        hand_xml.merge(obj_xml, merge_body=True)
        hand_xml.save_model(self.hand_obj_path)
        xml = hand_xml.get_xml()
        return xml

    def get_env_state(self):
        sim_state = self.sim.get_state()
        return sim_state

    def set_env_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def shift_hand(self, pos, quat=[1, 0, 0, 0], euler=None):
        joints_init = [0 for _ in range(20)]
        initial_pos = np.deg2rad(np.asarray(joints_init))
        if not euler:
            euler = Rotation.from_quat(quat).as_euler("xyz")
        sim_state = self.sim.get_state()
        sim_state.qpos[0: 3] = pos
        sim_state.qpos[3: 6] = euler
        sim_state.qpos[6: 26] = initial_pos
        self.sim.set_state(sim_state)
        self.sim.forward()


def eval_model(split="train"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Using GPUs " + os.environ["CUDA_VISIBLE_DEVICES"])

    dataset = GraspDataset(vis=False, split=split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    model = BackbonePointNet2().cuda()
    model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(os.path.join(f"{cfg['network']['eval']['model_path']}/model_008.pth")))
    model = model.eval()
    taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]
    success_dict = {}
    available_point = {}
    for tax in taxonomies:
        success_dict[tax] = 0
        available_point[tax] = 0
    for i, (data, index) in enumerate(dataloader):
        success_temp_dict = {}
        available_temp_point = {}
        for tax in taxonomies:
            success_temp_dict[tax] = 0
            available_temp_point[tax] = 0
        points = copy.deepcopy(data["point"])[0]
        img_id = index[0].numpy()
        if split == "val":
            print(f"{split.title()} scene id: {img_id - 16000}")
        else:
            print(f"{split.title()} scene id: {img_id}")
        xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets/new_hand_lifting.xml")
        env = MujocoEnv(split=split, vis=True)
        scene_xml = env.create_scene_obj("mujoco_exp/objects_xml", img_id)
        env.update_scene_model(scene_xml)
        state = env.get_env_state()
        init_obj_height = state.qpos[-5]

        for k in data:
            data[k] = data[k].cuda().float()
        bat_pred_graspable, bat_pred_pose, bat_pred_joint = model(data["point"], data["norm_point"].transpose(1, 2))
        pred_graspable, pred_pose, pred_joint = bat_pred_graspable[0], bat_pred_pose[0], bat_pred_joint[0]

        scene = trimesh.Scene()
        mesh, _, _ = scene_util.load_scene(img_id, split=split)
        scene.add_geometry(mesh)
        pc = trimesh.PointCloud(points, colors=cfg["color"]["pointcloud"])
        scene.add_geometry(pc)
        # scene.show()

        # success_rate_list = []

        for t in range(pred_graspable.size(-1)):
            tax = taxonomies[t]
            tax_gp, tax_pose, tax_joint = pred_graspable[:, :, t], pred_pose[:, :, t], pred_joint[:, :, t]
            out_gp = torch.argmax(tax_gp, dim=1).bool()
            if torch.sum(out_gp) > 0:
                out_score = F.softmax(tax_gp, dim=1)[:, 1]
                out_score = out_score[out_gp].detach().cpu().numpy()
                out_pose = tax_pose[out_gp].detach().cpu().numpy()
                out_joint = tax_joint[out_gp].detach().cpu().numpy()
                grasp_points = points[out_gp].detach().cpu().numpy()

                score_idx = np.argsort(out_score)[::-1]
                out_pose = out_pose[score_idx]
                out_joint = out_joint[score_idx]
                grasp_points = grasp_points[score_idx]

                hand_pose, hand_quat, hand_joint = [], [], []
                for j in range(grasp_points.shape[0]):
                    grasp_point = grasp_points[j]
                    depth, quat = out_pose[j][0], out_pose[j][1:]
                    depth -= 0.04
                    joint = out_joint[j]
                    pos, quat, joint = hand_util.cal_hand_param(depth, quat, joint, grasp_point, tax)
                    hand_pose.append(pos)
                    hand_quat.append(quat)
                    hand_joint.append(joint)
                hand_pose, hand_quat, hand_joint = np.asarray(hand_pose), np.asarray(hand_quat), np.asarray(hand_joint)

                topK = cfg["mujoco"]["topK"]
                if grasp_points.shape[0] > topK:
                    hand_pose, hand_quat, hand_joint = hand_pose[:topK], hand_quat[:topK], hand_joint[:topK]
                    available_temp_point[tax] = topK
                    available_point[tax] = available_point[tax] + topK
                else:
                    available_temp_point[tax] = grasp_points.shape[0]
                    available_point[tax] = available_point[tax] + grasp_points.shape[0]

                curr_success = 0

                for p, q in zip(hand_pose, hand_quat):
                    env.disable_gravity()
                    init_joint = np.asarray(grasp_dict_20f[tax]["joint_init"]) * np.pi / 180.
                    final_joint = np.asarray(grasp_dict_20f[tax]["joint_final"]) * np.pi / 180.
                    env.set_hand_pos(joint=init_joint, quat=q, pos=p)
                    env.step(100)
                    env.grasp(final_joint)
                    env.step(100)
                    env.lift()
                    env.step(500)
                    # env.open_gravity()

                    # for _ in range(200):
                    #     env.step(10)
                    #     curr_state = env.get_env_state().qpos
                    #     obj_height = curr_state[-5]
                    #     if obj_height > -0.05:
                    #         curr_success += 1
                    #         break

                    curr_state = env.get_env_state().qpos
                    obj_height = curr_state[-5]
                    if obj_height > -0.05:
                        curr_success += 1

                    env.set_env_state(state)

                # print(t1 - t0, "\t", t2 - t1, "\t", t3 - t2)
                success_temp_dict[tax] = curr_success
                success_dict[tax] = success_dict[tax] + curr_success

                # success_rate_list.append(success / ((i + 1) * 5.))

        # print("success rate:", success_rate_list)
        # glfw.destroy_window(env.viewer.window)

        curr_success_rate = []
        all_success_rate = []
        for tax in taxonomies:
            curr_success = success_temp_dict[tax]
            all_success = success_dict[tax]
            curr_point = available_temp_point[tax]
            all_point = available_point[tax]
            curr_success_rate.append(curr_success / curr_point if curr_point != 0 else 0)
            all_success_rate.append(all_success / all_point if all_point != 0 else 0)

        print("Current success rate: {} {} {} {} {} {} {} {} {} {}".format(taxonomies[0], curr_success_rate[0],
                                                                           taxonomies[1], curr_success_rate[1],
                                                                           taxonomies[2], curr_success_rate[2],
                                                                           taxonomies[3], curr_success_rate[3],
                                                                           taxonomies[4], curr_success_rate[4]))
        print("Whole success rate: {} {} {} {} {} {} {} {} {} {}".format(taxonomies[0], all_success_rate[0],
                                                                         taxonomies[1], all_success_rate[1],
                                                                         taxonomies[2], all_success_rate[2],
                                                                         taxonomies[3], all_success_rate[3],
                                                                         taxonomies[4], all_success_rate[4]))


if __name__ == "__main__":
    splits = ["train", "val", "test"]
    for split in splits:
        print(f"{OKGREEN}========== {split.upper()} DATASET EXPERIMENT =========={ENDC}")
        eval_model(split=split)
