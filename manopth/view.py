import torch
from manopth.manolayer import ManoLayer
import trimesh
import numpy as np

extrins = [[-0.9289850985569019, 0.28467625028492727, -0.2365293199223648, 0.3545380234718323],
           [-0.1923749691607686, 0.17456904958262143, 0.9656694663124715, -1.0083506107330322],
           [0.3161938612617523, 0.9425948649613648, -0.10740746086213025, 0.9894363880157471],
           [0, 0, 0, 1]]
extrins = np.asarray(extrins)

def main():
    # Initialize MANO layer
    ncomps = 45
    mano_layer = ManoLayer(use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    betas = torch.tensor([[0.6993994116783142,
                            -0.16909725964069366,
                            -0.8955091834068298,
                            -0.09764610230922699,
                            0.07754238694906235,
                            0.336286723613739,
                            -0.05547792464494705,
                            0.5248727798461914,
                            -0.38668063282966614,
                            -0.00133091164752841]])
    label = np.load('20200709-subject-01/20200709_141754/840412060917/labels_000071.npz')
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

    trans_obj = label['pose_y'][0]
    obj_mesh = trimesh.load(
        'models/002_master_chef_can/textured_simple.obj')
    # obj_mesh.export('002.OBJ')
    # exit()
    pose = np.vstack((trans_obj, np.array([[0, 0, 0, 1]], dtype=np.float32)))
    pose[1] *= -1
    pose[2] *= -1

    obj_mesh.apply_transform(pose)
    # mesh_mano.apply_transform(np.linalg.inv(pose))
    # mesh_mano.export("mano.stl")
    scene.add_geometry([mesh_mano])
    scene.add_geometry([obj_mesh])
    scene.show()


if __name__ == "__main__":
    main()
