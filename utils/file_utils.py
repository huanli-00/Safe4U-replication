import os
import logging


def safe_open(file_path, mode: str = "r", encoding="utf-8"):
    if not os.path.exists(file_path):
        file_dir, file_name = os.path.split(file_path)
        dir_check(file_dir)
        open(file_path, "a").close()
        # os.mknod(file_path)
    return open(file_path, mode, encoding=encoding)


def dir_check(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def copy_to(file, dst_dir):
    # copy file to dst_dir
    if not os.path.exists(file):
        logging.warning(f"FAIL to copy file: {file} not found, skip.")
        return
    file_dir, file_name = os.path.split(file)
    dst_file = os.path.join(dst_dir, file_name)
    dir_check(dst_dir)
    os.system(f"cp {file} {dst_file}")
    return dst_file
