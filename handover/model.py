import os
import sys
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
from utils.pointnet2_modules import PointnetFPModule, PointnetSAModule
import yaml

with open("/home/haonan/codes/handover/DLR-Handover/config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)


class BackbonePointNet2(nn.Module):
    def __init__(self):
        super(BackbonePointNet2, self).__init__()
        if cfg["network"]["train"]["use_norm_points"]:
            self.sa1 = PointnetSAModule(mlp=[3, 32, 32, 64], npoint=256, radius=0.05, nsample=32, bn=True)
        else:
            self.sa1 = PointnetSAModule(mlp=[0, 32, 32, 64], npoint=256, radius=0.05, nsample=32, bn=True)
        self.sa2 = PointnetSAModule(mlp=[64, 64, 64, 128], npoint=128, radius=0.1, nsample=64, bn=True)
        self.sa3 = PointnetSAModule(mlp=[128, 128, 128, 256], npoint=32, radius=0.2, nsample=128, bn=True)
        self.sa4 = PointnetSAModule(mlp=[256, 256, 256, 512], npoint=None, radius=None, nsample=None, bn=True)

        self.fp4 = PointnetFPModule(mlp=[768, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[384, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[320, 256, 128])
        if cfg["network"]["train"]["use_norm_points"]:
            self.fp1 = PointnetFPModule(mlp=[128 + 6, 128, 128])
        else:
            self.fp1 = PointnetFPModule(mlp=[128 + 3, 128, 128])

        # fc layer
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.channels = 27
        channels = self.channels * cfg["network"]["train"]["num_taxonomies"]
        self.conv3 = nn.Conv1d(128, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.gp_weight = torch.tensor(cfg["network"]["train"]["gp_weight"]).cuda().float()

    def forward(self, xyz, points):
        l1_xyz, l1_points = self.sa1(xyz.contiguous(), points.contiguous())
        l1_xyz_nan = torch.any(torch.isnan(l1_xyz))
        l1_points_nan = torch.any(torch.isnan(l1_points))
        l1_nan = torch.any(torch.tensor([l1_xyz_nan, l1_points_nan]))
        assert not l1_nan, "Nan values after self.sa1"

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_xyz_nan = torch.any(torch.isnan(l2_xyz))
        l2_points_nan = torch.any(torch.isnan(l2_points))
        l2_nan = torch.any(torch.tensor([l2_xyz_nan, l2_points_nan]))
        assert not l2_nan, "Nan values after self.sa2"

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_xyz_nan = torch.any(torch.isnan(l3_xyz))
        l3_points_nan = torch.any(torch.isnan(l3_points))
        l3_nan = torch.any(torch.tensor([l3_xyz_nan, l3_points_nan]))
        assert not l3_nan, "Nan values after self.sa3"

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        if not l4_xyz:
            l4_xyz_nan = False
        else:
            l4_xyz_nan = torch.any(torch.isnan(l4_xyz))
        l4_points_nan = torch.any(torch.isnan(l4_points))
        l4_nan = torch.any(torch.tensor([l4_xyz_nan, l4_points_nan]))
        assert not l4_nan, "Nan values after self.sa4"

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_points_nan = torch.any(torch.isnan(l3_points)).item()
        assert not l3_points_nan, "Nan values after self.fp4"

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_points_nan = torch.any(torch.isnan(l2_points)).item()
        assert not l2_points_nan, "Nan values after self.fp3"

        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l1_points_nan = torch.any(torch.isnan(l1_points)).item()
        assert not l1_points_nan, "Nan values after self.fp2"

        feature = self.fp1(xyz.contiguous(), l1_xyz, torch.cat((xyz.transpose(1, 2), points), dim=1), l1_points)
        feature_nan = torch.any(torch.isnan(feature)).item()
        assert not feature_nan, "Nan values after self.fp1"

        feature = F.leaky_relu(self.bn1(self.conv1(feature)), negative_slope=0.2)
        feature_nan = torch.any(torch.isnan(feature)).item()
        assert not feature_nan, "Nan values after F.leaky_relu + self.bn1 + self.conv1"

        feature = self.drop1(feature)
        feature_nan = torch.any(torch.isnan(feature)).item()
        assert not feature_nan, "Nan values after self.drop1"

        feature = F.leaky_relu(self.bn2(self.conv2(feature)), negative_slope=0.2)
        feature_nan = torch.any(torch.isnan(feature)).item()
        assert not feature_nan, "Nan values after F.leaky_relu + self.bn2 + self.conv2"

        feature = self.drop2(feature)
        feature_nan = torch.any(torch.isnan(feature)).item()
        assert not feature_nan, "Nan values after self.drop2"

        bat_pred = self.conv3(feature).permute(0, 2, 1).view(xyz.size(0), -1,
                                                             self.channels,
                                                             cfg["network"]["train"]["num_taxonomies"]).contiguous()
        bat_pred_nan = torch.any(torch.isnan(bat_pred)).item()
        assert not bat_pred_nan, "Nan values after self.conv3"

        pred_graspable, pred_pose, pred_joint = bat_pred[:, :, :2, :], bat_pred[:, :, 2:-20, :], bat_pred[:, :, -20:, :]

        return pred_graspable, pred_pose, pred_joint

    def get_loss(self, pred, data):
        gt_list = []
        tax_list = []
        bat_pred_graspable, bat_pred_pose, bat_pred_joint = pred
        for k in data:
            if data[k].size(-1) == 26:  # graspable 1, depth 1, quat 4, joint 20
                gt_list.append(data[k])
                tax_list.append(k)
        assert len(tax_list) == bat_pred_graspable.size(-1)
        loss_dict, acc_dict = {}, {}
        for i, gt in enumerate(gt_list):
            graspable, depth, quat, joint = gt[:, :, 0].long(), gt[:, :, 1], gt[:, :, 2:6], gt[:, :, 6:]
            pred_graspable, pred_depth, pred_quat, pred_joint = bat_pred_graspable[:, :, :, i], \
                                                                bat_pred_pose[:, :, 0, i], \
                                                                bat_pred_pose[:, :, 1:, i], \
                                                                bat_pred_joint[:, :, :, i]

            # print(f"Ground truth: graspable:{graspable}\t depth:{depth}\t quat:{quat}\t joint:{joint}")
            # print(f"Prediction: graspable:{pred_graspable}\t depth:{pred_depth}\t quat:{pred_quat}\t joint:{pred_joint}")

            gp_mask = torch.where(graspable > 0)

            # gp_loss = nn.CrossEntropyLoss(self.gp_weight)(pred_graspable.squeeze(), graspable.squeeze())
            gp_loss = nn.CrossEntropyLoss(self.gp_weight)(pred_graspable.view(-1, 2), graspable.view(-1))
            out_gp = torch.argmax(pred_graspable, dim=-1)
            gp_acc = self.two_class_acc(out_gp, graspable)

            if gp_mask[0].tolist():
                depth_loss = nn.SmoothL1Loss()(pred_depth[gp_mask], depth[gp_mask])
                depth_acc_mask = torch.abs(pred_depth[gp_mask] - depth[gp_mask]) < cfg["network"]["eval"]["dist_thresh"]
                depth_acc = torch.sum(depth_acc_mask) / float(pred_depth[gp_mask].size(0))
                rot_loss, rot_acc = QuatLoss()(pred_quat[gp_mask], quat[gp_mask])
            else:
                depth_loss = torch.tensor(0)
                depth_acc = torch.tensor(0)
                rot_loss, rot_acc = torch.tensor(0), torch.tensor(0)

            joint_loss = nn.MSELoss()(pred_joint, joint)

            loss_dict[tax_list[i]] = {
                "gp_loss": gp_loss,
                "depth_loss": depth_loss,
                "rot_loss": rot_loss,
                "joint_loss": joint_loss,
                "loss": gp_loss + depth_loss + rot_loss + joint_loss
            }

            acc_dict[tax_list[i]] = {
                "TP": gp_acc[0],
                "FP": gp_acc[1],
                "TN": gp_acc[2],
                "FN": gp_acc[3],
                "acc": gp_acc[4],
                "p": gp_acc[5],
                "r": gp_acc[6],
                "F1": gp_acc[7],
                "depth_acc": depth_acc.item(),
                "quat_acc": rot_acc.item(),
            }

        loss = 0
        for tax in tax_list:
            loss += (loss_dict[tax]["loss"])
        loss_dict["total_loss"] = loss
        return loss_dict, acc_dict

    def two_class_acc(self, out, gt):
        TP = torch.sum((out == 1) & (gt == 1)).float()
        FP = torch.sum((out == 1) & (gt == 0)).float()
        TN = torch.sum((out == 0) & (gt == 0)).float()
        FN = torch.sum((out == 0) & (gt == 1)).float()
        p = TP / (TP + FP) if TP + FP != 0 else torch.tensor(0)
        r = TP / (TP + FN) if TP + FN != 0 else torch.tensor(0)
        F1 = 1 * r * p / (r + p) if r + p != 0 else torch.tensor(0)
        acc = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else torch.tensor(0)
        acc_list = [TP.item(), FP.item(), TN.item(), FN.item(), acc.item(), p.item(), r.item(), F1.item()]
        return acc_list


class QuatLoss(nn.Module):
    def __init__(self):
        super(QuatLoss, self).__init__()

    def forward(self, pred_quat, gt_quat):
        position_loss = nn.SmoothL1Loss()(pred_quat, gt_quat)
        norm_pred_quat = F.normalize(pred_quat, dim=1)
        pred_R = self.quaternion_to_matrix(norm_pred_quat)
        gt_R = self.quaternion_to_matrix(gt_quat)
        cosine_angle = self.so3_relative_angle(pred_R, gt_R, cos_angle=True)
        angle_loss = -torch.sum(cosine_angle) / float(pred_R.size(0))
        quat_acc = torch.sum(cosine_angle >
                             np.cos(cfg["network"]["eval"]["quat_thresh"] / 180 * np.pi)) / float(pred_R.size(0))
        return angle_loss + position_loss, quat_acc

    def so3_relative_angle(self, R1, R2, cos_angle: bool = True):
        R12 = torch.bmm(R1, R2.permute(0, 2, 1))
        return self.so3_rotation_angle(R12, cos_angle=cos_angle)

    def so3_rotation_angle(self, R, eps: float = 1e-4, cos_angle: bool = True):
        N, dim1, dim2 = R.shape
        if dim1 != 3 or dim2 != 3:
            raise ValueError("Input has to be a batch of 3 x 3 Tensors.")

        rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

        # if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        #     raise ValueError("A matrix has trace outside valid range [-1 - eps, 3 + eps].")

        # clamp to valid range
        rot_trace = torch.clamp(rot_trace, -1.0 - eps, 3.0 + eps)

        # phi ... rotation angle
        phi = 0.5 * (rot_trace - 1.0)

        if cos_angle:
            return phi
        else:
            return phi.acos()

    def quaternion_to_matrix(self, quaternions):
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))


if __name__ == "__main__":
    data1 = torch.rand(1, 5000, 3).cuda()
    data2 = torch.rand(1, 5000, 3).permute(0, 2, 1).cuda()
    model = BackbonePointNet2().cuda()
    model(data1, data2)
