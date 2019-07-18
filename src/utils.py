import os

def all_files_under(folder, subfolder='images', endswith='.png'):
    new_folder = os.path.join(folder, subfolder)
    file_names =  [os.path.join(new_folder, fname)
                   for fname in os.listdir(new_folder) if fname.endswith(endswith)]

    return sorted(file_names)
