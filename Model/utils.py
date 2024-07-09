import random, json, shutil, os
import matplotlib.pyplot as plt

def visualize_slices(batch, idx1, idx2):
    images, labels = batch
    fig, axs = plt.subplots(2, 6, figsize=(15, 5))
    for i in range(6):
        axs[0, i].imshow(images[idx1][i][0], cmap='gray')
        axs[0, i].set_title(f'Image {idx1}, Slice {i+1}, Label: {labels[idx1]}')
    for i in range(6):
        axs[1, i].imshow(images[idx2][i][0], cmap='gray')
        axs[1, i].set_title(f'Image {idx2}, Slice {i+1}, Label: {labels[idx2]}')
    
    plt.tight_layout()
    plt.show()


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

    for class_dir in class_dirs:
        os.makedirs(os.path.join(data_dir, class_dir), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_dir), exist_ok=True)

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

    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        files = [f for f in os.listdir(class_path) if f.endswith('.nii.gz')]
        random.shuffle(files)
        test_class_path = os.path.join(test_dir, class_dir)
        
        num_files_to_move = int(len(files) * 0.2)
        
        for file in files[:num_files_to_move]:
            src_path = os.path.join(class_path, file)
            dest_path = os.path.join(test_class_path, file)
            shutil.move(src_path, dest_path)
            test_files[class_dir].append(file)
            print(f"Moved {src_path} to {dest_path}")

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