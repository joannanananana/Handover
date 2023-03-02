import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import os
import argparse
import tqdm
import trimesh
import numpy as np
from dataset import GraspDataset
import yaml
from utils import scene_util, common_util, hand_util
import copy
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
from model import BackbonePointNet2

with open("config/handover_config.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser("Handover model visualization")
parser.add_argument("--batchsize", type=int, default=cfg["network"]["train"]["batchsize"], help="input batch size")
parser.add_argument("--workers", type=int, default=cfg["network"]["train"]["workers"],
                    help="number of data loading workers")
parser.add_argument("--epoch", type=int, default=cfg["network"]["train"]["epochs"],
                    help="number of epochs for training")
parser.add_argument("--gpu", type=str, default=cfg["network"]["train"]["gpu"], help="specify gpu device")
parser.add_argument("--learning_rate", type=float, default=cfg["network"]["train"]["learning_rate"],
                    help="learning rate for training")
parser.add_argument("--optimizer", type=str, default=cfg["network"]["train"]["optimizer"], help="type of optimizer")
parser.add_argument("--model_path", type=str, default=cfg["network"]["eval"]["model_path"],
                    help="path of trained models")
FLAGS = parser.parse_args()

taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]


def vis_groundtruth(dataloader):
    for i, (data, index) in enumerate(dataloader):
        img_id = index[0].numpy()
        points = copy.deepcopy(data["point"])[0]
        tax_list = []
        gt_list = []
        for k in data:
            if data[k].size(-1) == 26:  # graspable 1, depth 1, quat 4, joint 20
                gt_list.append(data[k])
                tax_list.append(k)
        for j, gt in enumerate(gt_list):
            scene = trimesh.Scene()
            mesh, _, _ = scene_util.load_scene(img_id)
            scene.add_geometry(mesh)
            pc = trimesh.PointCloud(points, colors=cfg["color"]["pointcloud"])
            scene.add_geometry(pc)
            tax = tax_list[j]
            graspable, pose_label, joint_label = gt[0, :, 0], gt[0, :, 1:6], gt[0, :, 6:]
            grasp_points, pose, joints = points[graspable == 1], pose_label[graspable == 1], joint_label[graspable == 1]
            for k in range(len(grasp_points)):
                grasp_point = grasp_points[k].numpy()
                depth, quat = pose[k][0].numpy(), pose[k][1:].numpy()
                joint = joints[k].numpy()

                pos, quat, joint = hand_util.cal_hand_param(depth, quat, joint, grasp_point, tax)
                hand_mesh = scene_util.load_hand(pos, quat, joint, color=cfg["color"]["hand_mesh"])
                scene.add_geometry(hand_mesh)
                scene.show()
                break


def vis_model(model, dataloader, split):
    model = model.eval()
    for i, (data, index) in enumerate(dataloader):
        # points = copy.deepcopy(data["point"])[0].numpy()
        bat_points = copy.deepcopy(data["point"])
        bat_img_id = index.numpy()
        tax_list = []
        gt_list = []
        for k in data:
            if data[k].size(-1) == 26:
                gt_list.append(data[k])
                tax_list.append(k)
            data[k] = data[k].cuda().float()
        bat_pred_graspable, bat_pred_pose, bat_pred_joint = \
            model(data["point"], data["norm_point"].transpose(1, 2))
        # pred_graspable, pred_pose, pred_joint = bat_pred_graspable[0], bat_pred_pose[0], bat_pred_joint[0]

        # for points, pred_graspable, pred_pose, pred_joint, img_id in zip(bat_points, bat_pred_graspable, bat_pred_pose,
        #                                                                  bat_pred_joint, bat_img_id):
        for batch in range(FLAGS.batchsize):
            points, pred_graspable, pred_pose, pred_joint, img_id = bat_points[batch], bat_pred_graspable[batch], \
                                                                    bat_pred_pose[batch], bat_pred_joint[batch], bat_img_id[batch]
            points = points.numpy()
            for t in range(pred_graspable.size(-1)):  # for each taxonomy
                tax = taxonomies[t]
                if tax not in tax_list:
                    print(f"No available grasp taxonomy {tax} in ground truth")
                    continue
                scene = trimesh.Scene()
                mesh, _, _ = scene_util.load_scene(img_id, split=split)
                scene.add_geometry(mesh)
                pc = trimesh.PointCloud(points, colors=cfg["color"]["pointcloud"])
                scene.add_geometry(pc)

                tax_gp, tax_pose, tax_joint = pred_graspable[:, :, t], pred_pose[:, :, t], pred_joint[:, :, t]
                # socre = F.softmax(tax_gp, dim=1)[:, 1].detach().cpu().numpy()
                tax_gp, tax_pose, tax_joint = tax_gp.detach().cpu().numpy(), \
                                              tax_pose.detach().cpu().numpy(), tax_joint.detach().cpu().numpy()

                gt = gt_list[t][batch]
                gt_gp, gt_pose, gt_joint = gt[:, 0].numpy(), gt[:, 1:6].numpy(), gt[:, 6:].numpy()
                # gt_grasp_points, gt_pose, gt_joint = points[gt_gp == 1], gt_pose[gt_gp == 1], gt_joint[gt_gp == 1]

                out_gp = np.argmax(tax_gp, 1)
                if np.all(gt_gp == 0):
                    print(f"No suitable grasp configuration for {tax}!")
                    continue

                ava_grasp_points, ava_gt_pose, ava_gt_joint, ava_pred_pose, ava_pred_joint = [], [], [], [], []
                for j in range(points.shape[0]):
                    gp1, gp2 = gt_gp[j], out_gp[j]
                    if gp1 and gp2:
                        ava_grasp_points.append(points[j])
                        ava_gt_pose.append(gt_pose[j])
                        ava_gt_joint.append(gt_joint[j])
                        ava_pred_pose.append(tax_pose[j])
                        ava_pred_joint.append(tax_joint[j])

                ava_grasp_points, ava_gt_pose, ava_gt_joint, ava_pred_pose, ava_pred_joint = \
                    np.asarray(ava_grasp_points), np.asarray(ava_gt_pose), np.asarray(ava_gt_joint), \
                    np.asarray(ava_pred_pose), np.asarray(ava_pred_joint)

                # tax_grasp_points = points[out_gp == 1].detach().cpu().numpy()
                # tax_pose = tax_pose[out_gp == 1]
                # tax_joint = tax_joint[out_gp == 1]

                for j in range(ava_grasp_points.shape[0]):
                    grasp_point = ava_grasp_points[j]

                    gt_depth, gt_quat = ava_gt_pose[j][0], ava_gt_pose[j][1:]
                    gt_joint_config = ava_gt_joint[j]
                    gt_pos, gt_quat, gt_joint_config = hand_util.cal_hand_param(gt_depth, gt_quat, gt_joint_config, grasp_point, tax)
                    gt_hand_mesh = scene_util.load_hand(gt_pos, gt_quat, gt_joint_config, color=cfg["color"]["hand_mesh"])

                    pred_depth, pred_quat = ava_pred_pose[j][0], ava_pred_pose[j][1:]
                    pred_joint_config = ava_pred_joint[j]
                    pred_pos, pred_quat, pred_joint_config = hand_util.cal_hand_param(pred_depth, pred_quat, pred_joint_config, grasp_point, tax)
                    pred_hand_mesh = scene_util.load_hand(pred_pos, pred_quat, pred_joint_config, color=cfg["color"][tax])

                    grasp_pc = trimesh.PointCloud(grasp_point.reshape((-1, 3)), colors=cfg["color"]["grasp_point"])

                    scene.add_geometry([gt_hand_mesh, pred_hand_mesh, grasp_pc])
                    scene.show()
                    break


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    print("Using GPUs " + os.environ["CUDA_VISIBLE_DEVICES"])
    train_data = GraspDataset(split="train")
    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=FLAGS.batchsize,
                                                   shuffle=True,
                                                   num_workers=FLAGS.workers)
    val_data = GraspDataset(split="val")
    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=FLAGS.batchsize,
                                                 shuffle=True,
                                                 num_workers=FLAGS.workers)
    test_data = GraspDataset(split="test")
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=FLAGS.batchsize,
                                                  shuffle=False,
                                                  num_workers=FLAGS.workers)
    model = BackbonePointNet2().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(f"{cfg['network']['eval']['model_path']}/model_039.pth")))
    # vis_groundtruth(train_dataloader)
    vis_model(model, test_dataloader, split="test")
