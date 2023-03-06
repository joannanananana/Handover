from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import trimesh
import re
import multiprocessing as mp
import random
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
OKGREEN = "\033[92m"
ENDC = "\033[0m"


def ply2obj(root_dir, out_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split(".")[1] == "ply":
                file_path = os.path.join(root_dir, file_name)
                # print(file_path)
                mesh = trimesh.load_mesh(file_path)
                obj_name = file_path.split("/")[-1].split(".")[0] + ".obj"
                file_path_obj = os.path.join(out_dir, obj_name)
                mesh.export(file_path_obj)
                print(file_path_obj, "finish")


def obj2ply(root_dir, out_dir):
    for root, dirs, files in os.walk(root_dir):
        for i, file_name in enumerate(files):
            if file_name.split(".")[1] == "ply":
                file_path = os.path.join(root_dir, file_name)
                print(file_path)
                mesh = trimesh.load_mesh(file_path)
                ply_name = "obj_" + str(i).zfill(6) + ".ply"
                file_path_obj = os.path.join(out_dir, ply_name)
                mesh.export(file_path_obj)
                print(file_path_obj, "finish")


def ply2stl(root_dir, out_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split(".")[1] == "ply":
                file_path = os.path.join(root_dir, file_name)
                # print(file_path)
                mesh = trimesh.load_mesh(file_path)
                stl_name = file_path.split("/")[-1].split(".")[0] + ".stl"
                file_path_obj = os.path.join(out_dir, stl_name)
                mesh.export(file_path_obj)
                print(file_path_obj, "finish")


def vhacd(name_in, name_out):
    os.system("testVHACD --input {} --output {} --maxhulls 64 --concavity 0.0001 "
              "--gamma 0.0001 --maxNumVerticesPerCH 64 --resolution 5000000 --log log.txt".format(name_in, name_out))


def vhacd_to_piece(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if file_name.split(".")[1] == "obj" and "vhacd" in file_name:
                vhacd_name = file_name.split(".")[0]
                vhacd_path = os.path.join(root, file_name)
                vhacd_piece_path = os.path.join(root, vhacd_name)
                if not os.path.exists(vhacd_piece_path):
                    os.mkdir(vhacd_piece_path)
                meshes = trimesh.load_mesh(vhacd_path)
                mesh_list = meshes.split()
                for i, mesh in enumerate(mesh_list):
                    new_file_name = vhacd_name + "cvx_{}.stl".format(i)
                    new_file_path = os.path.join(vhacd_piece_path, new_file_name)
                    mesh.export(new_file_path)
                    print(new_file_path, "finish")
    # os.system(f"mv {os.path.join()}")


def create_xml(file_dir_name, root_dir, output_path):
    data = ET.Element("mujoco")
    data.set("model", file_dir_name[:10])
    compiler = ET.SubElement(data, "compiler")
    size = ET.SubElement(data, "size")
    compiler.set("angle", "radian")
    compiler.set("meshdir", "")
    compiler.set("texturedir", "")
    size.set("njmax", "500")
    size.set("nconmax", "100")
    item_asset = ET.SubElement(data, "asset")

    item_worldbody = ET.SubElement(data, "worldbody")
    item_body = ET.SubElement(item_worldbody, "body")
    item_body.set("name", file_dir_name[:10])
    item_body.set("pos", "0 0 0 ")
    item_body.set("euler", "0 0 0")

    item_joint_tx = ET.SubElement(item_body, "joint")
    item_joint_tx.set("name", file_dir_name[:10] + "_joint")
    item_joint_tx.set("type", "free")

    for root, dirs, files in os.walk(root_dir):
        for file_name in files:
            if "vhacd" not in file_name and file_name.split(".")[-1] == "stl":
                vhacd_piece_name = file_name.split(".")[0]
                item_mesh = ET.SubElement(item_asset, "mesh")
                item_mesh.set("name", "mesh_" + vhacd_piece_name)

                item_mesh.set("file", os.path.join(root, file_name))
                item_mesh.set("scale", "1 1 1")

                print(root_dir + "/" + vhacd_piece_name[:10] + ".png")

                item_geom = ET.SubElement(item_body, "geom")
                item_geom.set("pos", "0 0 0")
                item_geom.set("type", "mesh")
                item_geom.set("density", "0")
                item_geom.set("mesh", "mesh_" + vhacd_piece_name)
                item_geom.set("contype", "0")
                item_geom.set("conaffinity", "0")
                item_geom.set("group", "0")
                item_geom.set("friction", "1 0.005 0.0001")

            if "cvx" in file_name:
                vhacd_piece_name = file_name.split(".")[0]
                item_mesh = ET.SubElement(item_asset, "mesh")
                item_mesh.set("name", vhacd_piece_name)
                item_mesh.set("file", os.path.join(root, file_name))
                item_geom = ET.SubElement(item_body, "geom")
                item_geom.set("type", "mesh")
                item_geom.set("density", "2500")
                item_geom.set("mesh", vhacd_piece_name)
                item_geom.set("name", vhacd_piece_name)
                item_geom.set("group", "3")
                item_geom.set("condim", "4")
                item_geom.set("friction", "10")

    et = ET.ElementTree(data)
    fname = os.path.join(output_path, "{}.xml".format(file_dir_name))

    et.write(fname, encoding="utf-8", xml_declaration=True)
    x = minidom.parse(fname)
    with open(fname, "w") as f:
        f.write(x.toprettyxml(indent="  "))


if __name__ == "__main__":
    # splits = ["train", "test"]
    split = "test"
    # for split in splits:
    # step 1: ply to obj
    root_dir = os.path.join(dir_path, "../grasp_models")
    out_dir = os.path.join(dir_path, "../mujoco_exp/objects")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # ply2obj(root_dir, out_dir)
    # ply2stl(root_dir, out_dir)
    # print(f"{OKGREEN}========== STEP 1: PLY TO OBJ FINISHED! =========={ENDC}")

    # step 2: vhacd
    # pool = mp.Pool(10)
    # root_dir = out_dir
    # for root, dirs, files in os.walk(root_dir):
    #     for file_name in files:
    #         if file_name.split(".")[-1] == "obj":
    #             name_in = os.path.join(root, file_name)
    #             name_out = name_in.replace(".obj", "_vhacd.obj")
    #             if not os.path.exists(name_out):
    #                 pool.apply_async(vhacd, args=(name_in, name_out,))
    # pool.close()
    # pool.join()
    # print(f"{OKGREEN}========== STEP 2: VHACD FINISHED! =========={ENDC}")

    # step 3: vhacd to piece
    # vhacd_to_piece(root_dir)
    # files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    # for file_name in files:
    #     if file_name.split(".")[-1] == "stl":
    #         vhacd_name = file_name.split(".")[0] + "_vhacd"
    #         vhacd_path = os.path.join(root_dir, vhacd_name)
    #         stl_path = os.path.join(root_dir, file_name)
    #         os.system(f"mv {stl_path} {vhacd_path}")
    # print(f"{OKGREEN}========== STEP 3: VHACD TO PIECE FINISHED! =========={ENDC}")

    # step 4: create xml
    output_path = os.path.join(dir_path, "../mujoco_exp/objects_xml")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            src = os.path.join(root, dir_name)
            create_xml(dir_name, src, output_path)
            print("{} finish".format(src))
    print(f"{OKGREEN}========== STEP 4: CREATE XML FINISHED! =========={ENDC}")

    # # step 1: ply to obj and stl
    # root_dir = os.path.join(dir_path, "../temp_dataset_1")
    # out_dir = os.path.join(dir_path, "../temp_dataset/objects")
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    # ply2obj(root_dir, out_dir)
    # ply2stl(root_dir, out_dir)
    # print(f"{OKGREEN}========== STEP 1: PLY TO OBJ FINISHED! =========={ENDC}")
    #
    # # step 2: vhacd
    # pool = mp.Pool(10)
    # root_dir = out_dir
    # for root, dirs, files in os.walk(root_dir):
    #     for file_name in files:
    #         if file_name.split(".")[-1] == "obj":
    #             name_in = os.path.join(root, file_name)
    #             name_out = name_in.replace(".obj", "_vhacd.obj")
    #             if not os.path.exists(name_out):
    #                 pool.apply_async(vhacd, args=(name_in, name_out,))
    # pool.close()
    # pool.join()
    # print(f"{OKGREEN}========== STEP 2: VHACD FINISHED! =========={ENDC}")
    #
    # # step 3: vhacd to piece
    # vhacd_to_piece(root_dir)
    # files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    # for file_name in files:
    #     if file_name.split(".")[-1] == "stl":
    #         vhacd_name = file_name.split(".")[0] + "_vhacd"
    #         vhacd_path = os.path.join(root_dir, vhacd_name)
    #         stl_path = os.path.join(root_dir, file_name)
    #         os.system(f"mv {stl_path} {vhacd_path}")
    # print(f"{OKGREEN}========== STEP 3: VHACD TO PIECE FINISHED! =========={ENDC}")
    #
    # # step 4: create xml
    # output_path = os.path.join(dir_path, "../temp_dataset/objects_xml")
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # for root, dirs, files in os.walk(root_dir):
    #     for dir_name in dirs:
    #         src = os.path.join(root, dir_name)
    #         create_xml(dir_name, src, output_path)
    #         print("{} finish".format(src))
    # print(f"{OKGREEN}========== STEP 4: CREATE XML FINISHED! =========={ENDC}")
