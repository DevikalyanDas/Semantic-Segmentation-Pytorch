import  os
import shutil

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def unzip(src_path,dst_path):

    shutil.unpack_archive(src_path, dst_path)

def create_data_folder(file_path,data_path):

    t1 = 'leftImg8bit'
    t2 = 'gtFine'
    zip1 = t1 + '.zip'
    zip2 = t2 + '.zip'
    
    os.makedirs(file_path, exist_ok=True)
    os.makedirs(os.path.join(file_path,t1), exist_ok=True)
    os.makedirs(os.path.join(file_path,t2), exist_ok=True)
    
    print('Unpacking the data from .zip')    
    
    unzip(os.path.join(data_path, zip1),os.path.join(file_path,t1))
    unzip(os.path.join(data_path, zip2),os.path.join(file_path,t2))
    
    print('Data Folder ready for Dataset')
    