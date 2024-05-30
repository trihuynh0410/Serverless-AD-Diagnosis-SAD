import os
import shutil

root_dir = r""
dirs_to_delete = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file:
            src_file = os.path.join(dirpath, file)
            subject_dir = dirpath.split(os.sep)[9]
            dest_file = os.path.join(root_dir, subject_dir, file)
            remove = dirpath.split(os.sep)[10]
            del_dir = os.path.join(root_dir, subject_dir, remove)
            shutil.move(src_file, dest_file)
            dirs_to_delete.append(del_dir)
dirs_to_delete = list(set(dirs_to_delete))
for dir in dirs_to_delete:
    # print(dir)
    shutil.rmtree(dir)