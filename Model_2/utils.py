import os
import shutil
import json
import random

def count_files(directory,class_dirs):
    counts = {}
    for class_dir in class_dirs:
        class_path = os.path.join(directory, class_dir)
        if os.path.exists(class_path):
            files = [f for f in os.listdir(class_path) if f.endswith('.nii.gz')]
            counts[class_dir] = len(files)
        else:
            counts[class_dir] = 0
    return counts


def data_spliting(data_dir,test_dir,json_file_path):
    class_dirs = ['CN', 'MCI', 'Mild']
    test_files = {class_dir: [] for class_dir in class_dirs}

    # Create directories if they don't exist
    for class_dir in class_dirs:
        os.makedirs(os.path.join(data_dir, class_dir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)

    # Move files into respective class directories
    for filename in os.listdir(data_dir):
        if filename.endswith('.nii.gz') and os.path.isfile(os.path.join(data_dir, filename)):
            class_prefix = filename.split('_')[0]
            if class_prefix in class_dirs:
                src_path = os.path.join(data_dir, filename)
                dest_path = os.path.join(data_dir, class_prefix, filename)
                shutil.move(src_path, dest_path)
                print(f"Moved {src_path} to {dest_path}")
            else:
                print(f"Skipping {filename}, unknown class prefix")

    # Move 40 files from each class to the test directory
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        files = [f for f in os.listdir(class_path) if f.endswith('.nii.gz')]
        random.shuffle(files)
        test_class_path = os.path.join(test_dir, class_dir)
        
        for file in files[:40]:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(test_class_path, file)
            shutil.move(src_path, dest_path)
            test_files[class_dir].append(file)
            print(f"Moved {src_path} to {dest_path}")

    # Save the names of the test files in a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(test_files, json_file, indent=4)

    print(f"Test file names saved to {json_file_path}")

    print("File counts in /workspace/data:")
    data_counts = count_files(data_dir)
    for class_dir, count in data_counts.items():
        print(f"{class_dir}: {count}")

    print("\nFile counts in /workspace/test:")
    test_counts = count_files(test_dir)
    for class_dir, count in test_counts.items():
        print(f"{class_dir}: {count}")

    print("\nTotal counts:")
    for class_dir in class_dirs:
        total = data_counts[class_dir] + test_counts[class_dir]
        print(f"{class_dir}: {total}")

