"""
This file provides utils for ai_challenger dataset
"""
import json
import numpy as np
import os
import os.path
from PIL import Image
import parameters as pa
import sys

SET_A_IMG = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_images_20180103"
SET_A_LABEL = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_test_a_20180103/keypoint_test_a_annotations_20180103.json"
SET_B_IMG = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_test_b_20180103/keypoint_test_b_images_20180103"
SET_B_LABEL = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_test_b_20180103/keypoint_test_b_annotations_20180103.json"
SET_C_IMG = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911"
SET_C_LABEL = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_validation_20170911/keypoint_validation_annotations_20170911.json"
SET_D_IMG = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_train_20170909/keypoint_train_images_20170902"
SET_D_LABEL = "dataset/AI_challenger_keypoint/ai_challenger_keypoint_train_20170909/keypoint_train_annotations_20170909.json"
RESIZED_IMG_FOLDER = "dataset/gen/ai_challenger_ratio_kept"

AI_IPJC_FILE = "./dataset/gen/ai_ipjc.npy"
AI_INAME_FILE = "./dataset/gen/ai_iname.npy"

PH, PW = pa.PH, pa.PW
def resize_keep_ratio():
    
    label_collection = [SET_A_LABEL, SET_B_LABEL, SET_C_LABEL, SET_D_LABEL]
    img_folder_collection = [SET_A_IMG, SET_B_IMG, SET_C_IMG, SET_D_IMG]
    assert(len(label_collection) == len(img_folder_collection))
    assert(all([os.path.exists(f) for f in label_collection]) and all([os.path.exists(f) for f in img_folder_collection]))
    
    def resize_by_json(labels, img_folder):
        # drop images with more than 4 people appeared
        labels = [l for l in labels if len(l['keypoint_annotations'])<=pa.MAX_ALLOWED_PEOPLE]
        num_img = len(labels)
        ipjc_arr = np.zeros([num_img, pa.MAX_ALLOWED_PEOPLE, 6, 3], np.float32)
        iname_list = list()

        for idx, img_label in enumerate(labels):
            
            # Zoom Image
            name = img_label['image_id'] + ".jpg"
            file = os.path.join(img_folder, name)
            im = Image.open(file)
            np_im = np.asarray(im.convert("RGB"))
            ori_size = im.size
            ori_ratio = ori_size[0] / ori_size[1]
            if ori_ratio >= target_ratio:
                # Depends on width
                zoom_ratio = PW / ori_size[0]
                bg = np.zeros((int(ori_size[0] / target_ratio), ori_size[0], 3), np.uint8)

            elif ori_ratio < target_ratio:
                # Depends on height
                zoom_ratio = PH / ori_size[1]
                bg = np.zeros((ori_size[1], int(ori_size[1] * target_ratio), 3), np.uint8)
                
            bg[:ori_size[1], :ori_size[0], :] = bg[:ori_size[1], :ori_size[0], :] + np_im[:, :, :]
            re_im = Image.fromarray(bg, 'RGB')
            re_im = re_im.resize((PW, PH), Image.ANTIALIAS)
            out_path = os.path.join(RESIZED_IMG_FOLDER, name)
            re_im.save(out_path)
            print(str(idx) + ' ' + name)
            
            # Modify label
            iname_list.append(name)
            anno_dict = img_label['keypoint_annotations']
            for p, human_key in enumerate(img_label['keypoint_annotations']):
                anno = anno_dict[human_key]
                anno = np.asarray(anno, dtype=np.int32).reshape([14, 3])
                # annotation in ai challenger: 0/右肩，1/右肘，2/右腕，3/左肩，4/左肘，5/左腕
                visible = lambda ai_j: anno[ai_j, 2] == 1 # Looks for visible joints only
                def set_ipjc(pose_j, ai_j):
                    ipjc_arr[idx, p, pose_j, 0:2] = anno[ai_j, 0:2] * zoom_ratio
                    ipjc_arr[idx, p, pose_j, 2] = anno[ai_j, 2] # mask or visible are both 1 in annotation
                
                if visible(2): # Joint visibility
                    set_ipjc(0, 2)
                if visible(1):
                    set_ipjc(1, 1)
                if visible(0):
                    set_ipjc(2, 0)
                if visible(3):
                    set_ipjc(3, 3)
                if visible(4):
                    set_ipjc(4, 4)
                if visible(5):
                    set_ipjc(5, 5)
                
        iname_arr = np.asarray(iname_list)
        return [ipjc_arr, iname_arr]
        
    target_ratio = PW / PH
    
    files = [open(l) for l in label_collection]
    json_labels_list = [json.load(l) for l in files]
    [f.close() for f in files]
    
    la_im_list = list(zip(json_labels_list, img_folder_collection))
    ipjcs_inames_list = [resize_by_json(*la_im) for la_im in la_im_list] # [3][ipjc, iname]
    ipjc3, iname3 = list(zip(*ipjcs_inames_list))
    ipjc3 = np.reshape(np.asarray(ipjc3), [-1])
    iname3 = np.reshape(np.asarray(iname3), [-1])

    ipjc_con = np.concatenate(ipjc3, 0)
    iname_con = np.concatenate(iname3, 0)
    np.save(AI_IPJC_FILE, ipjc_con)
    np.save(AI_INAME_FILE, iname_con)

assert sys.version_info >= (3,5)
resize_keep_ratio()
    
    