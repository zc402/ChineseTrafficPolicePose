# from skimage.viewer import ImageViewer
import tensorflow as tf
import glob
import sys
import parameters as pa
import PAF_network
import numpy as np
import os
import cv2
from skimage.draw import line_aa


assert sys.version_info >= (3, 5)


def load_label(csv_file):
    """
    Label file is a csv file using number to mark gesture for each frame
    example content: 0,0,0,2,2,2,2,2,0,0,0,0,0
    :param csv_file:
    :return: list of int
    """
    with open(csv_file, 'r') as label_file:
        labels = label_file.read()

    labels = labels.split(",")
    labels = [int(l) for l in labels]
    # Labels is a list of int, representing gesture for each frame
    return labels

def random_video_clip(video, csv_file, frame_length):
    labels = load_label(csv_file)
    labels = np.array(labels, dtype=np.int64)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError("%s can't be opened by OpenCV" % video)
    v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_fps = int(cap.get(cv2.CAP_PROP_FPS))
    if v_fps != 15:
        raise ValueError("video %s have a frame rate of %d, not 15." % (video, v_fps))
    if v_size != len(labels):
        raise ValueError("Video and csv length not equal")
    start_ind = np.random.uniform(0, int(v_size-frame_length-1))
    # Skip this many frames
    for i in range(start_ind):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Unexpected end of file")
    frames = []
    for i in range(frame_length):
        ret, frame = cap.read()
        frames.append(frame)
    cap.release()
    frames = np.array(frames)

    labels = labels[start_ind: start_ind + frame_length]
    return labels, frames

def random_joints_clip(npy_joints, csv_file, frame_length)
    labels = load_label(csv_file)
    v_size = len(labels)
    labels = np.array(labels, dtype=np.int64)
    start_ind = np.random.uniform(0, int(v_size-frame_length-1))
    labels_cut = labels[start_ind:start_ind+frame_length]
    
    joints = np.load(npy_joints)
    joints_cut = joints[start_ind:start_ind+frame_length]
    return labels_cut, joints_cut

def labels_delay(labels, delay_frames):
    z = np.zeros((delay_frames), dtype=np.int64)
    l = len(labels)  # Original label len
    labels = np.concatenate((z, labels))  # len: delay + origion
    labels = labels[:l]
    return labels


def random_btjc_btl(batch_size, time_steps):
    """
    Load joint pos with labels at random time
    :param batch_size:
    :param time_steps:
    :return:
    """

    csv_list = glob.glob(os.path.join(pa.RNN_TRAIN_FOLDER, "*.csv"))
    list.shuffle(csv_list)

    btjc = []
    btl = []
    for b in range(batch_size):
        video = csv_list[b].replace(".csv", ".mp4")
        ls, fs = random_video_clip(video, csv_list[b], time_steps)
        ls = labels_delay(ls, pa.LABEL_DELAY_FRAMES)
        btl.append(ls)
        btjc.append(fs)




