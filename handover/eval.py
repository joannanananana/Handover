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
from dataset import GraspDataset
from model import BackbonePointNet2
import yaml
from utils import scene_util, common_util, hand_util, grasp_util, eval_util
from hitdlr_kinematics.hitdlr_layer.taxonomy_20dof import grasp_dict_20f
import trimesh
from dlr_mujoco_grasp import MujocoEnv
import glob
from multiprocessing import Pool


with open("config/handover_config.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]

OKGREEN = "\033[92m"
ENDC = "\033[0m"


def test(output_path="output", split="train"):
    model = BackbonePointNet2().cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(os.path.join(f"{cfg['network']['eval']['model_path']}/model_039.pth")))
    model.eval()

    dataset = GraspDataset(vis=False, split=split)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, (data, index) in enumerate(tqdm(dataloader)):
        points = copy.deepcopy(data["point"])[0].numpy()
        bat_points = copy.deepcopy(data["point"])
        bat_img_id = index.numpy()
        # img_id = index[0].numpy()
        # index = index.item()
        for k in data:
            data[k] = data[k].cuda().float()
        bat_pred_graspable, bat_pred_pose, bat_pred_joint = model(data["point"], data["norm_point"].transpose(1, 2))
        # print("bat_pred_graspable", bat_pred_graspable.shape)
        # print("bat_pred_pose", bat_pred_joint.shape)
        # print("bat_pred_joint", bat_pred_joint.shape)
        # pred_graspable, pred_pose, pred_joint = bat_pred_graspable[0], bat_pred_pose[0], bat_pred_joint[0]
        # print("pred_graspable", pred_graspable.shape)
        # print("pred_pose", pred_joint.shape)
        # print("pred_joint", pred_joint.shape)
        for batch in range(cfg["network"]["train"]["batchsize"]):
            points, pred_graspable, pred_pose, pred_joint, img_id = bat_points[batch], bat_pred_graspable[batch], \
                                                                    bat_pred_pose[batch], bat_pred_joint[batch], bat_img_id[batch]
            points = points.numpy()
            output_hand_grasp = {}
            img_id = img_id.item()
            for t in range(pred_graspable.size(-1)):
                tax = taxonomies[t]
                tax_gp, tax_pose, tax_joint = pred_graspable[:, :, t], pred_pose[:, :, t], pred_joint[:, :, t]
                # print("tax_gp", tax_gp.shape)
                # print("tax_pose", tax_pose.shape)
                # print("tax_joint", tax_joint.shape)
                out_gp = torch.argmax(tax_gp, dim=1).bool()
                # print("out_gp", out_gp.shape)
                if torch.sum(out_gp) > 0:
                    score = F.softmax(tax_gp, dim=1)[:, 1].detach().cpu().numpy()
                    # print("score", score.shape)
                    tax_gp, tax_pose, tax_joint = tax_gp.detach().cpu().numpy(), tax_pose.detach().cpu().numpy(), \
                                                  tax_joint.detach().cpu().numpy()
                    out_gp = np.argmax(tax_gp, 1)
                    # print("out_gp", out_gp.shape)
                    # exit()
                    out_score = score[out_gp == 1]
                    grasp_points = points[out_gp == 1]
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
                    out_pose, out_quat, out_joint = np.asarray(out_pose), np.asarray(out_quat), np.asarray(out_joint)

                    output_hand_grasp[tax] = {
                        "pos":          out_pose,
                        "quat":         out_quat,
                        "joint":        out_joint,
                        "score":        out_score
                    }
            np.save(os.path.join(output_path, f"img_{split}_{img_id}_grasp.npy"), output_hand_grasp)


def evaluate(root_path, img_id, split="train"):
    # print("img_id:", img_id)

    path = os.path.join(root_path, split)
    # scene = trimesh.Scene()
    obj_mesh, _, _ = scene_util.load_scene(img_id, split=split)
    # scene.add_geometry(obj_mesh)

    env = MujocoEnv(split=split, vis=False)
    scene_xml = env.create_scene_obj("mujoco_exp/objects_xml", img_id)
    env.update_scene_model(scene_xml)
    state = env.get_env_state()

    grasp = np.load(os.path.join(path, f"img_{split}_{img_id}_grasp.npy"), allow_pickle=True).item()
    if not grasp:
        print(f"{split} img id:", img_id, "No suitable grasp")
        # np.save(f"output_res/with_tax/{split}/img_{img_id}_res.npy", {})
        return

    penetration = {}
    complete_list = []
    for k in grasp:
        tax = k
        init_hand = trimesh.load(f"assets/dlr_init_hand/{tax}.stl")
        pos, quat, joint, score = grasp[tax]["pos"], grasp[tax]["quat"], grasp[tax]["joint"], grasp[tax]["score"]
        R = trimesh.transformations.quaternion_matrix(quat)
        if len(quat) == 1:
            R = R.reshape(1, 4, 4)
        R = R[:, :3, :3]
        depth_list, volume_list, success_list = [], [], []
        if len(pos) > 0:
            hand_pos, hand_R, hand_joint, _ = grasp_util.grasp_nms(pos, R, joint, score, None)

            # select at most 10 grasps for each taxonomy
            if len(pos) > 10:
                hand_pos, hand_R, hand_joint = hand_pos[:10], hand_R[:10], hand_joint[:10]

            hand_quat = common_util.matrix_to_quaternion(hand_R)

            for grasp_index, (p, q, j) in enumerate(zip(hand_pos, hand_quat, hand_joint)):  # 10 grasp configuration
                hand_mesh = scene_util.load_hand(p, q, j, color=cfg["color"][tax])
                depth, volume_sum = eval_util.calculate_metric(hand_mesh, obj_mesh)
                depth_list.append(depth)
                volume_list.append(volume_sum)
                env.disable_gravity()
                init_joint = np.asarray(grasp_dict_20f[tax]["joint_init"]) * np.pi / 180.
                final_joint = np.asarray(grasp_dict_20f[tax]["joint_final"]) * np.pi / 180.
                env.set_hand_pos(joint=init_joint, quat=q, pos=p)
                env.step(100)
                env.grasp(joint=final_joint)
                env.step(100)
                env.lift()
                env.step(500)

                curr_state = env.get_env_state().qpos
                obj_height = curr_state[-5]
                success = 1 if obj_height > -0.05 else 0
                env.set_env_state(state)
                success_list.append(success)

        mean_depth = np.mean(depth_list)
        mean_volume = np.mean(volume_list)
        mean_success = np.mean(success_list)
        complete_list.append(1) if np.any(success_list != 0) else complete_list.append(0)
        penetration[tax] = {
            "depth":        float(format(mean_depth, ".4f")),
            "volume":       float(format(mean_volume, ".4f")),
            "success":      float(format(mean_success, ".4f"))
        }
    completion = np.mean(complete_list)
    penetration["completion"] = completion
    print(f"{split} img id:", img_id, penetration)

    res_path = f"output_noise_res/with_tax/{split}"
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    file_path = os.path.join(res_path, f"img_{img_id}_res.npy")

    np.save(file_path, penetration)

    return penetration


def evaluate_without_tax(root_path, img_id, split="train"):
    # print("img_id:", img_id)

    path = os.path.join(root_path, split)
    scene = trimesh.Scene()
    obj_mesh, _, _ = scene_util.load_scene(img_id, split=split)

    env = MujocoEnv(split=split, vis=False)
    scene_xml = env.create_scene_obj("mujoco_exp/objects_xml", img_id)
    env.update_scene_model(scene_xml)
    state = env.get_env_state()

    grasp = np.load(os.path.join(path, f"img_{split}_{img_id}_grasp.npy"), allow_pickle=True).item()
    if not grasp:
        print(f"{split} img id:", img_id, "No suitable grasp")
        # np.save(f"output_res/without_tax/{split}/img_{img_id}_res.npy", {})
        return

    penetration = {}
    all_grasp = []
    for i, tax in enumerate(grasp.keys()):
        pos, quat, joint, score = grasp[tax]["pos"], grasp[tax]["quat"], grasp[tax]["joint"], grasp[tax]["score"]
        all_grasp.append(np.concatenate([pos, quat, joint, score[:, np.newaxis],
                                         np.asarray([i] * len(pos))[:, np.newaxis]], axis=-1))
    all_grasp = np.concatenate(all_grasp, axis=0)
    pos, quat, joint, score, taxs = all_grasp[:, :3], all_grasp[:, 3:7], \
                                    all_grasp[:, 7:27], all_grasp[:, 27], all_grasp[:, 28]
    R = trimesh.transformations.quaternion_matrix(quat)
    if len(quat) == 1:
        R = R.reshape(1, 4, 4)
    R = R[:, :3, :3]
    depth_list, volume_list, success_list = [], [], []
    if len(pos) > 0:
        hand_pos, hand_R, hand_joint, hand_tax = grasp_util.grasp_nms(pos, R, joint, score, taxs)

        # select at most 10 grasps for each taxonomy
        if len(pos) > 10:
            hand_pos, hand_R, hand_joint, hand_tax = hand_pos[:10], hand_R[:10], hand_joint[:10], hand_tax[:10]

        hand_quat = common_util.matrix_to_quaternion(hand_R)

        for grasp_index, (p, q, j, t) in enumerate(zip(hand_pos, hand_quat, hand_joint, hand_tax)):
            tax = taxonomies[int(t)]
            init_hand = trimesh.load(f"assets/dlr_init_hand/{tax}.stl")
            hand_mesh = scene_util.load_init_hand(p, q, init_hand, color=cfg["color"][tax])
            depth, volume_sum = eval_util.calculate_metric(hand_mesh, obj_mesh)
            env.disable_gravity()
            depth_list.append(depth)
            volume_list.append(volume_sum)
            init_joint = np.asarray(grasp_dict_20f[tax]["joint_init"]) * np.pi / 180.
            final_joint = np.asarray(grasp_dict_20f[tax]["joint_final"]) * np.pi / 180.
            env.set_hand_pos(joint=init_joint, quat=q, pos=p)
            env.step(100)
            env.grasp(joint=final_joint)
            env.step(100)
            env.lift()
            env.step(500)

            curr_state = env.get_env_state().qpos
            obj_height = curr_state[-5]
            success = 1 if obj_height > -0.05 else 0
            env.set_env_state(state)
            success_list.append(success)
    mean_depth = np.mean(depth_list)
    mean_volume = np.mean(volume_list)
    mean_success = np.mean(success_list)
    penetration = {
        "depth":        float(format(mean_depth, ".4f")),
        "volume":       float(format(mean_volume, ".4f")),
        "success":      float(format(mean_success, ".4f"))
    }
    print(f"{split} img id:", img_id, penetration)

    res_path = f"output_noise_res/without_tax/{split}"
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    file_path = os.path.join(res_path, f"img_{img_id}_res.npy")

    np.save(file_path, penetration)

    return penetration


def parallel_evaluate(root_path="output", split="train", proc=10, tax=True):
    if split == "train":
        imgIds = list(range(16000))
    elif split == "val":
        imgIds = list(range(16000, 20000))
    else:
        imgIds = list(range(2000))

    p = Pool(processes=proc)
    res_list = []
    for img_id in imgIds:
        if tax:
            res_list.append(p.apply_async(evaluate, (root_path, img_id, split,)))
        else:
            res_list.append(p.apply_async(evaluate_without_tax, (root_path, img_id, split,)))
    p.close()
    p.join()
    output = []
    for res in tqdm(res_list):
        output.append(res.get())
    return output


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print("Using GPUs " + os.environ["CUDA_VISIBLE_DEVICES"])
    splits = ["val", "test"]
    flags = [True, False]
    output_root_path = "output_noise"
    # for split in splits:
    #     output_path = os.path.join(output_root_path, split)
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #
    #     test(output_path=output_path, split=split)

    # exit()

    # output = parallel_evaluate(root_path=output_root_path, split="test", tax=True)
    # split = "train"
    # parallel_evaluate(output_root_path, split=split)

    # for using_tax in flags:
    #     for split in splits:
    #         parallel_evaluate(output_root_path, split=split, tax=using_tax)

    # parallel_evaluate(output_root_path, "test", tax=True)
    # output = parallel_evaluate(output_root_path, "val", tax=False)
    # output = parallel_evaluate(output_root_path, "test", tax=False)

    # imgIds = list(range(16000, 20000))
    # for img_id in imgIds:

    # evaluate(root_path=output_root_path, img_id=17689, split="val")

    # parallel_evaluate(root_path=output_root_path, split="test", tax=False)

    res_root_path = "output_noise_res"
    for using_tax in flags:
        s = "with" if using_tax else "without"
        s += "_tax"
        for split in splits:
            file_path = f"{res_root_path}/{s}/{split}"
            res = glob.glob(os.path.join(file_path, "*_res.npy"))
            output = [np.load(r, allow_pickle=True).item() for r in res]
            print(f"{OKGREEN}========== RESULT ON {split.upper()} {s.upper()} =========={ENDC}")
            if using_tax:
                res_dict = {}
                for tax in taxonomies:
                    res_dict[tax] = {}
                for tax in taxonomies:
                    res_dict[tax]["depth"] = []
                    res_dict[tax]["volume"] = []
                    res_dict[tax]["success"] = []
                for out in output:
                    for tax in taxonomies:
                        if tax in out:
                            res_dict[tax]["depth"].append(out[tax]["depth"])
                            res_dict[tax]["volume"].append(out[tax]["volume"])
                            res_dict[tax]["success"].append(out[tax]["success"])
                for tax in taxonomies:
                    print(tax + ":")
                    print("\tmean success:", np.mean(res_dict[tax]["success"]))
                    print("\tmean depth:", np.mean(res_dict[tax]["depth"]))
                    print("\tmean volume:", np.mean(res_dict[tax]["volume"]))
            else:
                mean_depth = np.mean([out["depth"] for out in output])
                mean_volume = np.mean([out["volume"] for out in output])
                mean_success = np.mean([out["success"] for out in output])
                print("\tmean success:", mean_success)
                print("\tmean depth:", mean_depth)
                print("\tmean volume:", mean_volume)

    # imgIds = list(range(16000, 20000))
    # for img_id in imgIds:
    #     evaluate_without_tax(output_root_path, img_id=img_id, split="val")
