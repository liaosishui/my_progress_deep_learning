import os
import zipfile

def unzip_files(src_path, target_path):
    if os.path.exists(src_path) == False:
        print('src_path is not exist.')
        return
    if os.path.exists(os.path.join(target_path, 'mask_detection')) == False:
        zf = zipfile.ZipFile(src_path, 'r')
        zf.extractall(path=target_path)
        zf.close()