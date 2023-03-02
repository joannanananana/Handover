import trimesh
import os
import yaml

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "./config/handover_config.yaml"), "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]
init_path = "assets/dlr_init_hand"
final_path = "assets/dlr_final_hand"

quat = [0.9155427, 0.3815801, 0.1271934, 0]
rotation = trimesh.transformations.quaternion_matrix(quat)

for tax in taxonomies:
    init_hand_path = os.path.join(init_path, tax + ".stl")
    init_hand = trimesh.load_mesh(init_hand_path)
    init_hand.visual.face_colors = cfg["color"][tax]
    # init_hand.apply_transform(rotation)
    init_hand.show()

    final_hand_path = os.path.join(final_path, tax + ".stl")
    final_hand = trimesh.load_mesh(final_hand_path)
    final_hand.visual.face_colors = cfg["color"][tax]
    # final_hand.apply_transform(rotation)
    final_hand.show()
