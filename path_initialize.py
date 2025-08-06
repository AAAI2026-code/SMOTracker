import os
import shutil


PATH_CHECK_LIST={
    "buffer":["video_images","video_output","videos"],
    "data":[],
    "weights":[]
}

if __name__ == '__main__':
    for dir_name in PATH_CHECK_LIST.keys():
        content_dir_list=PATH_CHECK_LIST[dir_name]
        os.makedirs(dir_name,exist_ok=True)
        for content_dir_name in content_dir_list:
            content_dir_path=os.path.join(dir_name,content_dir_name)
            os.makedirs(content_dir_path,exist_ok=True)
