import os
import shutil


train_data = "./train_obj"
test_data = "./test_obj"
label = "./picked_obj"

dataset_path = "backup/dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Generate train data
train_data_path = os.path.join(dataset_path, "train/data")
train_label_path = os.path.join(dataset_path, "train/label")
if not os.path.exists(train_label_path):
    os.makedirs(train_data_path)
if not os.path.exists(train_label_path):
    os.makedirs(train_label_path)

print("===== Start processing train data =====")
for i, file in enumerate(sorted(os.listdir(train_data), key=lambda x: os.path.splitext(x)[0].lower())):
    file_name = os.path.splitext(file)[0]
    print(f"Index: {str(i).zfill(3)}, Object: {file_name}", end="     ")
    obj_name = file_name + ".obj"
    label_name = file_name + ".npy"
    obj_name_new = "obj_" + str(i).zfill(6) + ".obj"
    label_name_new = "obj_" + str(i).zfill(6) + ".npy"
    shutil.copyfile(os.path.join(train_data, obj_name), os.path.join(train_data_path, obj_name_new))
    shutil.copyfile(os.path.join(label, label_name), os.path.join(train_label_path, label_name_new))
    print("Finish!")


# Generate train data
test_data_path = os.path.join(dataset_path, "test/data")
test_label_path = os.path.join(dataset_path, "test/label")
if not os.path.exists(test_label_path):
    os.makedirs(test_data_path)
if not os.path.exists(test_label_path):
    os.makedirs(test_label_path)

print("===== Start processing test data =====")
for i, file in enumerate(sorted(os.listdir(test_data), key=lambda x: os.path.splitext(x)[0].lower())):
    file_name = os.path.splitext(file)[0]
    print(f"Index: {str(i).zfill(3)}, Object: {file_name}", end="     ")
    obj_name = file_name + ".obj"
    label_name = file_name + ".npy"
    obj_name_new = "obj_" + str(i).zfill(6) + ".obj"
    label_name_new = "obj_" + str(i).zfill(6) + ".npy"
    shutil.copyfile(os.path.join(test_data, obj_name), os.path.join(test_data_path, obj_name_new))
    shutil.copyfile(os.path.join(label, label_name), os.path.join(test_label_path, label_name_new))
    print("Finish!")
