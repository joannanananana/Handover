import os

data_path = "dataset/train/data"
label_path = "dataset/train/label"

for folder in sorted(os.listdir(data_path)):
    folder_path = os.path.join(data_path, folder)
    indices = []
    for point in sorted(os.listdir(folder_path)):
        name = int(point[:6])
        indices.append(name)
    if indices == list(range(1000)):
        continue
    else:
        print("current path:", folder_path)

print()

for folder in sorted(os.listdir(label_path)):
    folder_path = os.path.join(label_path, folder)
    indices = []
    for point in sorted(os.listdir(folder_path)):
        name = int(point[:6])
        indices.append(name)
    if indices == list(range(1000)):
        continue
    else:
        print("current path:", folder_path)
