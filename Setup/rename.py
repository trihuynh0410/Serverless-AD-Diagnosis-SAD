import os
base_dir = r''  
stage = ["CN", "MCI", "Mild"]
for s in stage:
    s_dir = os.path.join(base_dir, s)
    list_dir = os.listdir(s_dir)
    for id in list_dir:
        id_dir = os.path.join(s_dir, id)
        for dirpath, dirnames, filenames in os.walk(id_dir):
            for file in filenames:
                if file.endswith('.nii'):
                    new_id = s + "_" + id
                    old_file_path = os.path.join(dirpath, file)
                    new_file_path = os.path.join(dirpath, f'{new_id}.nii')
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed {old_file_path} to {new_file_path}")
