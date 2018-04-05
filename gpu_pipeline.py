import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import os
import math
import random
from tensorflow.python import debug as tf_debug

def part_confidence_maps(ipjc_tensor, paf_mask, heat_size):
    """
    Part Confidence Maps
    :param ipjc_tensor: [Image] [Person] [Joint] [cx,cy]
    :param paf_mask: [Image] [Person] [Joint]
    :param heat_size: [W,H]
    :return: [Image] [Joint] [H] [W]
    """

    assert(len(ipjc_tensor.get_shape().as_list()) == 4)
    assert(ipjc_tensor.get_shape().as_list()[3] == 2)
    assert(len(paf_mask.get_shape().as_list()) == 3)
    variance = 1.1
    cx = ipjc_tensor[:, :, :, 0:1]
    cy = ipjc_tensor[:, :, :, 1:2]

    w = tf.range(heat_size[0], dtype=tf.float32)
    h = tf.range(heat_size[1], dtype=tf.float32)
    w = tf.square(w - cx) # 4-d [I] [P] [J] [W]
    h = tf.square(h - cy) # 4-d [I] [P] [J] [H]
    # Distance square have 5-d(01234), ind3 and ind4 are height and width
    # Insert a new empty dim at ind3
    w = tf.expand_dims(w, 3) # [i] [p] [j] [1] [W]
    h = tf.expand_dims(h, 4) # [i] [p] [j] [H] [1]
    dist_sq = w + h # Distances with each pixel
    # Dist_sq now has shape of [i:p:j:H,W]
    exponent = dist_sq / 2.0 / variance / variance
    heatmap = tf.exp(-exponent)
    paf_mask = tf.expand_dims(tf.expand_dims(paf_mask, 3), 3)
    
    heatmap = heatmap * paf_mask
    # Combine heatmaps of same joint from different person
    heatmap = tf.reduce_sum(heatmap, axis=1)
  
    return heatmap


def joint_to_bone(ipjc_tensor, pcm_mask):
    """
    Convert pcm tensor to paf tensor
    :param ipjc_tensor: [Image] [Person] [Joint] [cx,cy]
    :param pcm_mask: [Image] [Person] [Joint]
    :return: ipbpc_tensor, paf_mask
    """
    # I P B c1, Bone = 0,1,2,3,4
    ipbp1c = ipjc_tensor[:, :, 0:5, :]
    # I P B c2, Bone = 1,2,3,4,5
    ipbp2c = ipjc_tensor[:, :, 1:6, :]
    # I P B (p1,p2) c
    ipbpc_tensor = tf.stack([ipbp1c, ipbp2c], axis=3)
    # When either one of joints is 0, bone is masked
    paf_mask = tf.multiply(pcm_mask[:, :, 0:5] ,pcm_mask[:, :, 1:6], name='paf_mask')
    
    return ipbpc_tensor, paf_mask
    

def part_affinity_fields(ipbpc_tensor, paf_mask, heat_size, line_width = 2.5):
    """
    Part Affinity Fields
    :param ipbpc_tensor: [Image] [Person] [Bone] [p1,p2] [x,y]
    :param paf_mask: [Image] [Person] [Bone]
    :param heat_size: [W,H]
    :param line_width: width of each bones
    :return: PAF [Image] [Bone] [H] [W] [horizontal, vertical]
    """

    # [Image] [Person] [Bone] [p1,p2] [1] [1] [x,y] for broadcasting
    ipbpc_tensor = tf.expand_dims(tf.expand_dims(ipbpc_tensor, 4), 4)
    # [Image] [Person] [Bone] [1] [1] [x,y]
    p1 = ipbpc_tensor[:, :, :, 0, :, :, :]
    # p1p2: [I] [P] [B] [1] [1] [x,y]
    p1p2 = ipbpc_tensor[:, :, :, 1, :, :, :] - ipbpc_tensor[:, :, :, 0, :, :, :]
    # p1p2_norm: [I] [P] [B] [1] [1]
    p1p2_norm = tf.norm(p1p2, axis=5) + 1e-7 # Avoid dividing by 0
    # Index matrices
    index_wh = tf.meshgrid(*[tf.range(length) for length in heat_size])
    p3 = tf.stack(index_wh, axis=2, name='form_meshgrid') # [H] [W] [x,y]
    p3 = tf.cast(p3, tf.float32)
    p1p3 = tf.subtract(p3, p1) # [I] [P] [B] [H] [W] [x,y]
    # Cross Product : U x V = Ux * Vy - Uy * Vx  p1p2 x p1p3, IPBWH
    cross_product = p1p2[:, :, :, :, :, 0] * p1p3[:, :, :, :, :, 1] - p1p2[:, :, :, :, :, 1] * p1p3[:, :, :, :, :, 0]
    # Perpendicular distance
    d = tf.abs(cross_product / p1p2_norm)
    # Dot Product
    dot_product = p1p2[:, :, :, :, :, 0] * p1p3[:, :, :, :, :, 0] + p1p2[:, :, :, :, :, 1] * p1p3[:, :, :, :, :, 1]
    # Projection Length
    p = tf.divide(dot_product ,p1p2_norm, name="dot_product") # Dot product with sign
    half_line_width = line_width / 2.
    # [I] [P] [B] [W] [H]
    valid_d_mask = tf.less(d, half_line_width, name="d_mask")
    valid_p_mask = tf.logical_and(tf.less(p, p1p2_norm + half_line_width), tf.greater(p, -half_line_width), name="p_mask")
    valid_mask = tf.logical_and(valid_d_mask, valid_p_mask, name="dp_mask")
    valid_mask = tf.expand_dims(valid_mask, axis=5) # IPBHW[1]
    valid_mask = tf.cast(valid_mask, tf.float32)
    p1p2_unit = tf.divide(p1p2, tf.expand_dims(p1p2_norm, axis=5), name="p1p2_unit") # unit on heatmap IPBHW[xy]
    paf_heatmaps_each_person = tf.multiply(valid_mask, p1p2_unit, name="heat_dp_mask") # IPBHW[xy]
    # Mask unused dimensions: IPB111
    paf_mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(paf_mask, 3), 3), 3)
    paf_heatmaps_each_person = tf.multiply(paf_heatmaps_each_person, paf_mask, name="heat_paf_mask")
    paf_heatmaps = tf.reduce_sum(paf_heatmaps_each_person, axis=1) # IBHW[xy]

    return paf_heatmaps

def pcm_paf_to_nhwc(pcm, paf):
    """
    Pack picture pcm paf together
    :param pcm: IJHW
    :param paf: IBHWV
    :return: NHWC (?, H, W, 6+5*2)
    """
    nhwc_pcm = tf.transpose(pcm, [0,2,3,1]) # NHWC
    nhwbv_paf = tf.transpose(paf, [0,2,3,1,4])
    s = nhwbv_paf.get_shape().as_list()
    nhwc_paf = tf.reshape(nhwbv_paf, [-1, s[1], s[2], s[3] * s[4]]) # NHWC
    nhwc = tf.concat([nhwc_pcm, nhwc_paf], axis=3)
    return nhwc


def input_images_augment_nhwc(img_tensor, nhwc_tensor):
    """
    Input image augmentation
    :param img_tensor: [Image][H][W][3]
    :param nhwc_tensor: [Image][H][W][Joint,Bone*2]
    :return: img, nhwc
    """
    image_num = nhwc_tensor.get_shape().as_list()[0]
    # Rotate
    radian = tf.random_uniform([image_num], -10., 10., dtype=tf.float32) * math.pi / 180
    nhwc_tensor = tf.contrib.image.rotate(nhwc_tensor, radian, "BILINEAR")
    img_tensor = tf.contrib.image.rotate(img_tensor, radian, "BILINEAR")
    return img_tensor, nhwc_tensor


def build_training_pipeline(ipjc_holder, img_holder):
    """
    Build the data preparation pipeline
    :param ipjc_holder: [Image][Person][Joint][x,y,mask]
    :param img_holder: [Image][H][W][C]
    :return: [I][H][W][3], [Image][H][W][Joint,Bone*2]
    """
    with tf.variable_scope("prepare"):
        HEAT_SIZE = (82, 47)
        ipjc_tensor = tf.divide(ipjc_holder[:, :, :, 0:2], 8)  # TODO: img to heat, divide 8 in ipjc.npy
        pcm_mask = ipjc_holder[:, :, :, 2]
    with tf.variable_scope("pcm"):
        pcm = part_confidence_maps(ipjc_tensor, pcm_mask, HEAT_SIZE)
    with tf.variable_scope("jtob"):
        ipbpc_tensor, paf_mask = joint_to_bone(ipjc_tensor, pcm_mask)
    with tf.variable_scope("paf"):
        paf = part_affinity_fields(ipbpc_tensor, paf_mask, HEAT_SIZE)
    with tf.variable_scope("con"):
        nhwc = pcm_paf_to_nhwc(pcm, paf)
    with tf.variable_scope("aug"):
        img_aug, nhwc_aug = input_images_augment_nhwc(img_holder, nhwc)
        return img_aug, nhwc_aug
    
def training_samples_gen(batch_size):
    """
    Generate training samples
    :param batch_size:
    :return: img_batch, ipjc_batch
    """
    # TODO: use both MPII and AI Dataset
    # RESIZED_RATIO_KEPT = "./dataset/gen/ratio_kept"
    # IPJC_FILE = "./dataset/gen/ipjc.npy" # [Image][Person][Joint][x,y,mask]
    # INAME_FILE = "./dataset/gen/iname.npy"
    
    # AI dataset
    RESIZED_RATIO_KEPT = "dataset/gen/ai_challenger_ratio_kept"
    IPJC_FILE = "./dataset/gen/ai_ipjc.npy"  # [Image][Person][Joint][x,y,mask]
    INAME_FILE = "./dataset/gen/ai_iname.npy"
    
    ipjc = np.load(IPJC_FILE)
    iname = np.load(INAME_FILE)
    
    ipjc_list = ipjc.tolist()
    iname_list = iname.tolist()
    zipped_list = list(zip(iname_list, ipjc_list)) # [I](name, ipjc)
    
    while True:
        # Shuffle
        random.shuffle(zipped_list)
        iname_list_s, ipjc_list_s = zip(*zipped_list) # Unzip
        for num in range(0, len(iname_list_s) - batch_size, batch_size):
            img_batch = [np.asarray(Image.open(os.path.join(RESIZED_RATIO_KEPT, name)), np.float32) / 255.
                        for name in iname_list_s[num : num+batch_size]]
            ipjc_batch = ipjc_list_s[num : num+batch_size]
            yield img_batch, ipjc_batch
    

