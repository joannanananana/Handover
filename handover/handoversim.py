import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import copy
from model import BackbonePointNet2
import yaml
from utils import scene_util, common_util, hand_util, grasp_util, eval_util
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import trimesh
import glob
from multiprocessing import Pool


with open("./config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]

OKGREEN = "\033[92m"
ENDC = "\033[0m"

parser = argparse.ArgumentParser("Handover Simulation")
parser.add_argument("--vis", "-v", action="store_true", default=False)
args = parser.parse_args()


def generate_grasps(vis):
    model = BackbonePointNet2().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join("experiment/20220308_201539_quat_loss8/checkpoints/model_079.pth")))
    model.eval()

    extrinsics = np.load("/home/haonan/codes/manopth/extrinsics.npy")

    mano_mesh = trimesh.load_mesh("/home/haonan/codes/manopth/mano.stl")
    hand_points = mano_mesh.vertices
    choice = np.random.choice(len(hand_points), 256, replace=True)
    hand_points = hand_points[choice]

    points = np.load("/home/haonan/codes/manopth/points_temp.npy")
    center = np.mean(points, axis=0)
    norm_points = points - center
    cloud = np.concatenate([points, norm_points], axis=-1)
    choice = np.random.choice(len(cloud), 256, replace=True)
    selected_points = points[choice]
    cloud = torch.tensor(cloud[choice]).unsqueeze(0).cuda().float()

    bat_pred_graspable, bat_pred_pose, bat_pred_joint = model(cloud[:, :, :3], cloud[:, :, 3:].transpose(1, 2))
    pred_graspable, pred_pose, pred_joint = bat_pred_graspable[0], bat_pred_pose[0], bat_pred_joint[0]

    all_grasp = []
    for t in range(pred_graspable.size(-1)):
        tax = taxonomies[t]
        tax_gp, tax_pose, tax_joint = pred_graspable[:, :, t], pred_pose[:, :, t], pred_joint[:, :, t]
        out_gp = torch.argmax(tax_gp, dim=1).bool()
        if torch.sum(out_gp) > 0:
            score = F.softmax(tax_gp, dim=1)[:, 1].detach().cpu().numpy()
            tax_gp, tax_pose, tax_joint = tax_gp.detach().cpu().numpy(), tax_pose.detach().cpu().numpy(), \
                                          tax_joint.detach().cpu().numpy()
            out_gp = np.argmax(tax_gp, 1)
            out_score = score[out_gp == 1]
            grasp_points = selected_points[out_gp == 1]
            tax_pose = tax_pose[out_gp == 1]
            tax_joint = tax_joint[out_gp == 1]

            out_pose, out_quat, out_joint = [], [], []
            for j in range(grasp_points.shape[0]):
                grasp_point = grasp_points[j]
                depth, quat = tax_pose[j][0], tax_pose[j][1:]
                depth -= 0.04
                joint = tax_joint[j]
                pos, quat, joint = hand_util.cal_hand_param(depth, quat, joint, grasp_point, tax)
                out_pose.append(pos)
                out_quat.append(quat)
                out_joint.append(joint)
            out_pose = np.asarray(out_pose)
            out_quat = np.asarray(out_quat)
            out_joint = np.asarray(out_joint)
            all_grasp.append(np.concatenate([out_pose, out_quat, out_joint, out_score[:, np.newaxis],
                                             np.asarray([t] * len(out_pose))[:, np.newaxis]], axis=-1))
    all_grasp = np.concatenate(all_grasp, axis=0)

    pos, quat, joint, score, taxs = all_grasp[:, :3], all_grasp[:, 3:7], all_grasp[:, 7:27], all_grasp[:, 27], all_grasp[:, 28]
    R = trimesh.transformations.quaternion_matrix(quat)
    if len(quat) == 1:
        R = R.reshape(1, 4, 4)
    R = R[:, :3, :3]
    if len(pos) > 0:
        hand_pos, hand_R, hand_joint, hand_tax = grasp_util.grasp_nms(pos, R, joint, score, taxs)
        hand_tax = hand_tax.astype(int)
        hand_quat = common_util.matrix_to_quaternion(hand_R)
        for p, q, j, t in zip(hand_pos, hand_quat, hand_joint, hand_tax):
            tax = taxonomies[t]
            hand_mesh = scene_util.load_hand(p, q, j, color=cfg["color"][tax])
            # r = trimesh.transformations.euler_matrix(np.pi, 0, 0)
            vis = args.vis
            if vis:
                scene = trimesh.Scene()
                obj_points = trimesh.PointCloud(points, colors=cfg["color"]["object"])
                scene.add_geometry([obj_points, hand_mesh])
                scene.show()
                # exit()


    # dataset = GraspDataset(vis=False, split=split)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    #
    # for i, (data, index) in enumerate(tqdm(dataloader)):
    #     points = copy.deepcopy(data["point"])[0].numpy()
    #     bat_points = copy.deepcopy(data["point"])
    #     bat_img_id = index.numpy()
    #     # img_id = index[0].numpy()
    #     # index = index.item()
    #     for k in data:
    #         data[k] = data[k].cuda().float()
    #     bat_pred_graspable, bat_pred_pose, bat_pred_joint = model(data["point"], data["norm_point"].transpose(1, 2))
    #     # print("bat_pred_graspable", bat_pred_graspable.shape)
    #     # print("bat_pred_pose", bat_pred_joint.shape)
    #     # print("bat_pred_joint", bat_pred_joint.shape)
    #     # pred_graspable, pred_pose, pred_joint = bat_pred_graspable[0], bat_pred_pose[0], bat_pred_joint[0]
    #     # print("pred_graspable", pred_graspable.shape)
    #     # print("pred_pose", pred_joint.shape)
    #     # print("pred_joint", pred_joint.shape)
    #     for batch in range(cfg["network"]["train"]["batchsize"]):
    #         points, pred_graspable, pred_pose, pred_joint, img_id = bat_points[batch], bat_pred_graspable[batch], \
    #                                                                 bat_pred_pose[batch], bat_pred_joint[batch], bat_img_id[batch]
    #         points = points.numpy()
    #         output_hand_grasp = {}
    #         img_id = img_id.item()
    #         for t in range(pred_graspable.size(-1)):
    #             tax = taxonomies[t]
    #             tax_gp, tax_pose, tax_joint = pred_graspable[:, :, t], pred_pose[:, :, t], pred_joint[:, :, t]
    #             # print("tax_gp", tax_gp.shape)
    #             # print("tax_pose", tax_pose.shape)
    #             # print("tax_joint", tax_joint.shape)
    #             out_gp = torch.argmax(tax_gp, dim=1).bool()
    #             # print("out_gp", out_gp.shape)
    #             if torch.sum(out_gp) > 0:
    #                 score = F.softmax(tax_gp, dim=1)[:, 1].detach().cpu().numpy()
    #                 # print("score", score.shape)
    #                 tax_gp, tax_pose, tax_joint = tax_gp.detach().cpu().numpy(), tax_pose.detach().cpu().numpy(), \
    #                                               tax_joint.detach().cpu().numpy()
    #                 out_gp = np.argmax(tax_gp, 1)
    #                 # print("out_gp", out_gp.shape)
    #                 # exit()
    #                 out_score = score[out_gp == 1]
    #                 grasp_points = points[out_gp == 1]
    #                 tax_pose = tax_pose[out_gp == 1]
    #                 tax_joint = tax_joint[out_gp == 1]
    #
    #                 out_pose, out_quat, out_joint = [], [], []
    #                 for j in range(grasp_points.shape[0]):
    #                     grasp_point = grasp_points[j]
    #                     depth, quat = tax_pose[j][0], tax_pose[j][1:]
    #                     depth -= 0.04
    #                     joint = tax_joint[j]
    #                     pos, quat, joint = hand_util.cal_hand_param(depth, quat, joint, grasp_point, tax)
    #                     out_pose.append(pos)
    #                     out_quat.append(quat)
    #                     out_joint.append(joint)
    #                 out_pose, out_quat, out_joint = np.asarray(out_pose), np.asarray(out_quat), np.asarray(out_joint)
    #
    #                 output_hand_grasp[tax] = {
    #                     "pos":          out_pose,
    #                     "quat":         out_quat,
    #                     "joint":        out_joint,
    #                     "score":        out_score
    #                 }
    #         np.save(os.path.join(output_path, f"img_{split}_{img_id}_grasp.npy"), output_hand_grasp)

if __name__ == "__main__":
    generate_grasps(vis=True)
