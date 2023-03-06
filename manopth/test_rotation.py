import trimesh
import numpy as np

# matrix = trimesh.transformations.euler_matrix(np.pi / 4, 0, -np.pi / 4)
# # print(trimesh.transformations.quaternion_from_euler(45, 0, -45))
# rads = trimesh.transformations.euler_from_matrix(matrix, "rxyz")
# print(rads)
# # print(np.deg2rad(45))
# # print(np.arctan(np.sqrt(2) / 2))

extrins = [[-0.9289850985569019, 0.28467625028492727, -0.2365293199223648, 0.3545380234718323],
           [-0.1923749691607686, 0.17456904958262143, 0.9656694663124715, -1.0083506107330322],
           [0.3161938612617523, 0.9425948649613648, -0.10740746086213025, 0.9894363880157471],
           [0, 0, 0, 1]]
extrins = np.asarray(extrins)

r = trimesh.transformations.euler_matrix(np.pi, 0, 0)

print(trimesh.transformations.euler_from_matrix(extrins.dot(r), axes="rxyz"))

# r = np.array([[ 0.20556626,  0.9745685 ,  0.08921175,  0.20227359],
#        [ 0.8555502 , -0.13470192, -0.49988922, -0.16387235],
#        [-0.4751593 ,  0.17908548, -0.86148244,  0.75131875]])


# matrix = trimesh.transformations.euler_from_matrix(r[:, :3], "rxyz")
# print(matrix)
