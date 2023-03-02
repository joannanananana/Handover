import trimesh
import os

taxonomies = ["Parallel_Extension", "Pen_Pinch", "Palmar_Pinch", "Precision_Sphere", "Large_Wrap"]


if __name__ == "__main__":
    for tax in taxonomies:
        init_hand_path = os.path.join("./assets/dlr_init_hand", tax + ".stl")
        scene = trimesh.Scene()
        init_hand_mesh = trimesh.load_mesh(init_hand_path)
        scene.add_geometry(init_hand_mesh)
        scene.show()
