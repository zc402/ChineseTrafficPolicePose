import os
from os import path
import pickle
import json
import random
import bidirectional_resize as bir
import numpy as np

def load_aicha_to_list(folder_path):
    # Use cache
    if not os.path.exists("./_cache"):
        os.mkdir("./_cache")
    if os.path.exists("./_cache/label.bin"):
        with open("./_cache/label.bin", "rb") as f:
            print("Read labels from cache")
            return pickle.load(f)

    # Load from file
    label_folders = ["train", "test_a", "test_b", "val"]
    list_PA = []  # list: path,annotation
    for f in label_folders:
        set_folder = path.join(folder_path, f)  # (train test val) set
        # Check set folder existence
        if not path.exists(set_folder):
            raise FileNotFoundError("Folder " + f + " not found")
        # Extract image labels as: list [path, content]
        annotations = path.join(set_folder, "annotations.json")
        with open(annotations, "r") as file_ann:
            json_list = json.load(file_ann)  # json_list: annotation list
        for json_img in json_list:  # json_img: annotation for 1 image
            name = json_img["image_id"] + ".jpg"
            # Image full path
            full_path = path.join(set_folder, "images")
            full_path = path.join(full_path, name)
            # Check image existence
            if not path.exists(full_path):
                raise FileNotFoundError("Image " + full_path + " not found")
            list_PA.append((full_path, json_img))

    print("Read " + str(len(list_PA)) + " image labels.")
    with open("./_cache/label.bin", "wb+") as f:
        pickle.dump(list_PA, f)
    return list_PA

# load_aicha_to_list("/media/zc/Ext4-1TB/AI_challenger_keypoint")


def _heatmap_1pt(x, y, width, height, variance):
    "Draw heatmap of 1 point"
    one = np.ones([height, width], dtype=np.float32)
    # background: [h][w][2:xy]
    horizontal = np.arange(width)
    horizontal = horizontal[np.newaxis, :]  # h,w
    horizontal = horizontal * one
    vertical = np.arange(height)
    vertical = vertical[:, np.newaxis]  # h,w
    vertical = vertical * one
    hwc = np.stack([horizontal, vertical], axis=-1)  # hwc: h,w,c  c:xy

    # point: [h][w][xy]
    one = np.ones([height, width, 2])
    pt_hwc = np.array([x, y])
    pt_hwc = pt_hwc[np.newaxis, np.newaxis, :]
    pt_hwc = pt_hwc * one

    # norm
    distance_2 = hwc - pt_hwc  # 2: xy
    distance_2 = distance_2.astype(np.float32)
    norm = np.linalg.norm(distance_2, axis=2, keepdims=False)
    exp = np.exp(-(norm / 2.0 / variance / variance))
    return exp


def anno_resize(anno, resize_wh, resize_record):
    """
    Resize the x,y value in an annotation, in place.
    :param anno:
    :param resize_wh:
    :param resize_record:
    :return: None
    """
    w, h = resize_wh
    if "keypoint_annotations" not in anno:
        raise ValueError(str(anno) + " is not an image label from AI_Challenger dataset.")
    ks = anno["keypoint_annotations"]
    for k in ks.keys():  # k: One human
        # Resize x,y, leave "visible" alone
        p_xyv = ks[k]  # x, y, visible. vi=1可见，vi=2不可见，vi=3不在图内或不可推测
        p_xyv = np.asarray(p_xyv, dtype=np.int)
        p_xyv = p_xyv.reshape([14, 3])
        for j in range(14):
            xyv = p_xyv[j]
            nx, ny = bir.resize_pt(xyv[0:2], resize_record)
            xyv[0] = int(nx)
            xyv[1] = int(ny)
        p_xyv = np.reshape(p_xyv, [-1])
        p_xyv = list(p_xyv)
        ks[k] = p_xyv

def heatmap_label(label, img_wh, zoom_times):
    """Return heatmap JWH"""
    img_w, img_h = img_wh
    img_w, img_h = (img_w//zoom_times, img_h//zoom_times)
    if "keypoint_annotations" not in label:
        raise ValueError(str(label) + " is not an image label from AI_Challenger dataset.")
    ks = label["keypoint_annotations"]
    heat_pjwh = []
    for k in ks.keys():  # k: One human
        p_xyv = ks[k]  # x, y, visible. vi=1可见，vi=2不可见，vi=3不在图内或不可推测
        p_xyv = np.asarray(p_xyv, dtype=np.float32)
        p_xyv = p_xyv.reshape([14, 3])
        heat_jwh = []
        for j in range(14):  # each joint
            x,y,v = p_xyv[j]
            x,y = (x//zoom_times, y//zoom_times)
            if v==0 or v==1:
                heat = _heatmap_1pt(x, y, img_w, img_h, 1.8)
            else:
                heat = np.zeros([img_h, img_w], dtype=np.float32)
            heat_jwh.append(heat)
        heat_jwh = np.stack(heat_jwh, axis=0)
        heat_pjwh.append(heat_jwh)
    heat_pjwh = np.stack(heat_pjwh, axis=0)
    heat_max_jwh = np.amax(heat_pjwh, axis=0)  # Sum each person's joint heatmap
    return heat_max_jwh



