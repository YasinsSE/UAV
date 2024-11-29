import argparse
import os
import shutil
from random import sample

# Add Parser
parser = argparse.ArgumentParser()

parser.add_argument("--train", type=int, default=90, help="Percentage of train set")
parser.add_argument("--validation", type=int, default=7, help="Percentage of validation set")
parser.add_argument("--test", type=int, default=3, help="Percentage of test set")
parser.add_argument("--folder", type=str, default="C:/Users/Yasins/Desktop/UAV/dataset", help="Folder that contains images and labels")
parser.add_argument("--dest", type=str, default="C:/Users/Yasins/Desktop/UAV/splitted_dataset", help="Destination folder for split dataset")

args = parser.parse_args()


def get_difference_from_2_list(list1, list2):
    return list(set(list1).difference(set(list2)))


def get_split_data(list_id):
    n_train = (len(list_id) * args.train) // 100
    n_valid = (len(list_id) * args.validation) // 100

    train = sample(list_id, n_train)
    remaining = get_difference_from_2_list(list_id, train)
    valid = sample(remaining, n_valid)
    test = get_difference_from_2_list(remaining, valid)

    return train, valid, test


def make_folder():
    folders = ["images", "labels"]
    inner_folders = ["train", "val", "test"]

    if not os.path.isdir(args.dest):
        os.mkdir(args.dest)

    for folder in folders:
        path = os.path.join(args.dest, folder)
        if not os.path.isdir(path):
            os.mkdir(path)

        for in_folder in inner_folders:
            inner_path = os.path.join(path, in_folder)
            if not os.path.isdir(inner_path):
                os.mkdir(inner_path)


def copy_image(file, id_folder):
    inner_folders = ["train", "val", "test"]

    # Copy image
    source = os.path.join(args.folder, file)
    dest_folder = os.path.join(args.dest, 'images', inner_folders[id_folder])
    shutil.copy(source, dest_folder)

    # Copy label
    label_file = file.rsplit(".", 1)[0] + ".txt"
    label_src = os.path.join(args.folder, label_file)
    label_dest = os.path.join(args.dest, 'labels', inner_folders[id_folder])
    shutil.copy(label_src, label_dest)


if args.train + args.validation + args.test != 100:
    print("Total percentage must equal 100%. Exiting.")
    exit()

image_files = [file for file in os.listdir(args.folder) if file.endswith((".jpg", ".png"))]
list_id = list(range(len(image_files)))

train, valid, test = get_split_data(list_id)
make_folder()

for count, file in enumerate(image_files):
    if count in train:
        copy_image(file, 0)
    elif count in valid:
        copy_image(file, 1)
    else:
        copy_image(file, 2)
