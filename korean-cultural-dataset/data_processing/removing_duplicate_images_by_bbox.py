import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def read_json(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_image_files(building_dir):
    image_counter = 0
    for file in os.listdir(building_dir):
        try:
            extention = file.split(".")[1]
            if extention != "json":
                image_counter += 1
        except:
            return file

    return image_counter


def get_box_size(boxcorners):

    x_1 = boxcorners[0]
    y_1 = boxcorners[1]
    x_2 = boxcorners[2]
    y_2 = boxcorners[3]

    box_size = abs(x_2 - x_1) * abs(y_2 - y_1)
    return box_size


def get_box_info(building_dir):
    box_info = {}

    json_files = sorted([f for f in os.listdir(building_dir) if f.endswith(".json")])

    for file in json_files:
        json_file_dir = os.path.join(building_dir, file)
        annotation = read_json(json_file_dir)
        boxcorners = annotation["regions"][0]["boxcorners"]

        file_id = file.split(".")[0]

        box_size = get_box_size(boxcorners)
        box_info[file_id] = box_size

    return box_info


def select_ids(box_info: dict):
    ids = np.array(list(box_info.keys()))
    box_sizes = np.array(list(box_info.values()))
    selected_ids = []
    counts, bins, patches = plt.hist(box_sizes, bins=10, edgecolor="black")
    plt.close()

    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            bin_indices = np.where((box_sizes >= bins[i]) & (box_sizes <= bins[i + 1]))[0]
        else:
            bin_indices = np.where((box_sizes >= bins[i]) & (box_sizes < bins[i + 1]))[0]
        num_values_to_select = int(100 / (len(bins) - 1))
        if len(bin_indices) > num_values_to_select: 
            selected_indices = np.random.choice(bin_indices, num_values_to_select, replace=False)
        else:
            selected_indices = bin_indices 
        selected_ids.extend(ids[selected_indices])

    return selected_ids


def pick_files_in_dir(unzip_dir):
    root = os.listdir(unzip_dir)
    for building in root:
        building_dir = os.path.join(unzip_dir, building)
        if not os.path.isdir(building_dir):
            print(f"Skip (Not a directory): {building_dir}")
            continue

        try:
            box_info = get_box_info(building_dir)
        except Exception as e:
            print(f"Error in get_box_info for {building_dir}: {e}")
            continue
        keep_file_ids = select_ids(box_info)

        for file in os.listdir(building_dir):
            file_name = file.split(".")[0]
            if file_name not in keep_file_ids:
                os.remove(os.path.join(building_dir, file))

        print(f"Left files in {building_dir} : ", len(os.listdir(building_dir)) // 2)


LOG_FILE = "./processed_dirs.log"


def load_processed_dirs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return set(f.read().splitlines())
    return set()


def save_processed_dir(directory):
    with open(LOG_FILE, "a") as f:
        f.write(directory + "\n")


def main(args):
    processed_dirs = load_processed_dirs()
    all_dirs = [
        os.path.join(args.root_dir, d)
        for d in os.listdir(args.root_dir)
        if os.path.isdir(os.path.join(args.root_dir, d))
    ]

    new_dirs = [d for d in all_dirs if d not in processed_dirs]
    skipped_dirs = [d for d in all_dirs if d in processed_dirs]

    if skipped_dirs:
        print("\n[Skipped directories] (already processed):")
        for skipped in skipped_dirs:
            print(f"pass: {skipped}")
    for unzip_dir in new_dirs:
        print(f"\n[Processing directory]: {unzip_dir}")
        pick_files_in_dir(unzip_dir)
        print("\nProcessing complete!")
        save_processed_dir(unzip_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean building image filtering")
    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to the root directory to process"
    )
    args = parser.parse_args()
    main(args)
