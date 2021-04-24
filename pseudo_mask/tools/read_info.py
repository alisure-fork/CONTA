import os
from alisuretool.Tools import Tools


def read_image_info(image_info_root):
    image_info_path = os.path.join(image_info_root, "deal", "image_info_list_change_person2.pkl")
    # image_info_list = Tools.read_from_pkl(image_info_path)[::200]
    image_info_list = Tools.read_from_pkl(image_info_path)
    return image_info_list

