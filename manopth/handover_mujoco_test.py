import mujoco_py
from mujoco_py import MjSim, MjViewer, MjRenderContextOffscreen
import torch
from manopth.manolayer import ManoLayer
import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET
import copy
from xml.dom import minidom
from mujoco_points import generatePointCloud
import yaml
import shutil


LABEL_DIR = "20200709-subject-01"
CAMERA_SERIAL = '932122062010'
CALIBRATION_DIR = "calibration"
WORLD_CAMERA = "840412060917"
XML_DIR = "mujoco_exp/objects_xml"
OBJ_STL_DIR = "mujoco_exp/objects"
# POIN

OBJ_DICT = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

with open("config/handoversim.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

def get_config(config_file):
    calibration_extrinsic = config_file["extrinsics"]
    mano_calib = config_file["mano_calib"][0]
    object_id = config_file["ycb_ids"]
    grasp_object_index = config_file["ycb_grasp_ind"]
    num_frames = config_file["num_frames"]
    mano_sides = config_file["mano_sides"][0]

    return calibration_extrinsic, mano_calib, object_id, grasp_object_index, num_frames, mano_sides


def get_camera_extrinsic(calibration_extrinsic):
    file_path = "extrinsics_" + calibration_extrinsic
    extrinsic_config_path = os.path.join(cfg["calibration_path"], file_path, "extrinsics.yml")
    with open(extrinsic_config_path, "r") as f:
        camera_cfg = yaml.load(f, Loader=yaml.FullLoader)
        extrinsics = camera_cfg["extrinsics"][cfg["selected_camera"]]
        extrinsics = np.asarray(extrinsics).reshape((3, 4))
        extrinsics = np.vstack([extrinsics, [0, 0, 0, 1]])
        return extrinsics


def get_mano_mesh(mano_calib, num_frames, data_path, grasp_object_index, object_id, mano_sides, save=False, vis=False):
    ncomps = 45
    mano_layer = ManoLayer(use_pca=True, ncomps=ncomps, flat_hand_mean=False, side=mano_sides)
    file_path = "mano_" + mano_calib
    mano_config_path = os.path.join(cfg["calibration_path"], file_path, "mano.yml")
    with open(mano_config_path, "r") as f:
        mano_cfg = yaml.load(f, Loader=yaml.FullLoader)
        betas = mano_cfg["betas"]
        betas = torch.tensor([betas])
    label_path = os.path.join(data_path, cfg["world_camera"])
    label_name = "labels_" + str(num_frames - 1).zfill(6) + ".npz"
    label = np.load(os.path.join(label_path, label_name))

    pose_m = label['pose_m']

    scene = trimesh.Scene()

    pose = torch.from_numpy(pose_m)
    vert, _ = mano_layer(pose[:, 0:48], betas, pose[:, 48:51])
    vert /= 1000
    vert = vert.view(778, 3)
    vert = vert.numpy()
    vert[:, 1] *= -1
    vert[:, 2] *= -1
    faces = mano_layer.th_faces.numpy()
    mesh_mano = trimesh.Trimesh(vertices=vert, faces=faces)

    trans_obj = label['pose_y'][grasp_object_index]
    obj_name = cfg["obj_dict"][object_id[grasp_object_index]]
    obj_path = os.path.join("models", obj_name, "textured_simple.obj")
    obj_mesh = trimesh.load(obj_path)

    pose = np.vstack((trans_obj, np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose[1] *= -1
    pose[2] *= -1

    # obj_mesh.apply_transform(pose)
    mesh_mano.apply_transform(np.linalg.inv(pose))
    # mesh_mano.export("mano.stl")
    scene.add_geometry([mesh_mano])
    scene.add_geometry([obj_mesh])
    if vis:
        scene.show()

    if save:
        mesh_mano.export("mano.stl")

    return label


def write_xml(extrinsics, grasp_object_index, object_id, label):
    object_name = cfg["obj_dict"][object_id[grasp_object_index]]
    object_num = object_name[:3]
    xml_in_path = os.path.join(cfg["xml_path"], object_num + "_vhacd.xml")
    tree = ET.parse(xml_in_path)
    root = tree.getroot()

    e1_file = os.path.join(cfg["object_path"], object_num + ".stl")
    e1 = ET.Element("mesh", {"file": e1_file, "name": object_num + "_stl"})
    e2_file = "mano.stl"
    e2 = ET.Element("mesh", {"file": e2_file, "name": "hand"})
    root[2].append(e1)
    root[2].append(e2)

    trans_obj = label['pose_y'][grasp_object_index]
    trans_obj = np.vstack((trans_obj, np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pos = "{} {} {}".format(trans_obj[0][3], trans_obj[1][3], trans_obj[2][3])
    euler = trimesh.transformations.euler_from_matrix(trans_obj, axes="rxyz")
    euler = "{} {} {}".format(euler[0], euler[1], euler[2])
    root[3][0].set("euler", euler)
    root[3][0].set("pos", pos)

    common_attrib = {"condim": "4", "density": "2500", "friction": "10", "group": "0", "type": "mesh"}
    e3_attrib = copy.deepcopy(common_attrib)
    e3_attrib["mesh"] = object_num + "_stl"
    e3_attrib["name"] = object_num + "_stl"
    e3_attrib["rgba"] = "0 0 0 1"
    e3 = ET.Element("geom", e3_attrib)
    e4_attrib = copy.deepcopy(common_attrib)
    e4_attrib["mesh"] = "hand"
    e4_attrib["name"] = "hand"
    e4_attrib["rgba"] = "1 1 1 1"
    e4 = ET.Element("geom", e4_attrib)
    root[3][0].append(e3)
    root[3][0].append(e4)

    r = trimesh.transformations.euler_matrix(np.pi, 0, 0)
    floor_euler = trimesh.transformations.euler_from_matrix(extrinsics.dot(r), "rxyz")
    floor_euler = "{} {} {}".format(floor_euler[0], floor_euler[1], floor_euler[2])
    floor = ET.Element("geom", {"type": "plane", "size": "2 2 1", "pos": "0 0 -0.5", "euler": floor_euler,
                                "name": "floor", "rgba": "1 1 1 1"})
    camera_pos = "{} {} {}".format(extrinsics[0][3], extrinsics[1][3], extrinsics[2][3])
    camera = ET.Element("camera", {"fovy": "45", "pos": camera_pos, "euler": floor_euler, "name": "camera"})
    light = ET.Element("light", {"pos": "0 -2 0"})
    root[3].append(floor)
    root[3].append(camera)
    root[3].append(light)

    xml_out_path = "temp.xml"
    tree.write(xml_out_path, encoding="utf-8", xml_declaration=True)
    x = minidom.parse(xml_out_path)
    with open(xml_out_path, "w") as f:
        f.write(x.toprettyxml(indent="  "))

    extrinsics_out_path = "extrinsics.npy"
    np.save(extrinsics_out_path, extrinsics)


def handover_mujoco(vis=False):
    model = mujoco_py.load_model_from_path("temp.xml")
    sim = MjSim(model)
    viewer = MjViewer(sim)
    model.opt.gravity[-1] = 0
    camera_viewer = MjRenderContextOffscreen(sim, 0)

    object_points = generatePointCloud(sim, camera_viewer, vis)
    points_save_path = "points_temp.npy"
    np.save(points_save_path, object_points)

    # os.system("python /home/haonan/codes/handover/DLR-Handover/handoversim.py --vis")


if __name__ == "__main__":
    for file in os.listdir(cfg["label_path"]):
        data_path = os.path.join(cfg["label_path"], file)
        # data_path = '20200709-subject-01/20200709_141754/'
        with open(os.path.join(data_path, "meta.yml"), "r") as f:
            meta_cfg = yaml.load(f, Loader=yaml.FullLoader)
            calibration_extrinsic, mano_calib, object_id, grasp_object_index, num_frames, mano_sides = get_config(meta_cfg)
            extrinsics = get_camera_extrinsic(calibration_extrinsic)
            label = get_mano_mesh(mano_calib, num_frames, data_path, grasp_object_index, object_id, mano_sides, save=True, vis=True)
            write_xml(extrinsics, grasp_object_index, object_id, label)
            handover_mujoco(vis=False)
            # exit()

