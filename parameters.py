"""
Global parameters
"""
import os
import glob
import numpy as np

PH, PW = (512, 512)  # Size of network input picture
HEAT_ZOOMING_RATE = 8
assert(PW % HEAT_ZOOMING_RATE == 0 and PH % HEAT_ZOOMING_RATE == 0)
HEAT_H, HEAT_W = PW // HEAT_ZOOMING_RATE, PH // HEAT_ZOOMING_RATE  # Size of heatmap
HEAT_SIZE = (HEAT_W, HEAT_H)  # 64, 64
MAX_ALLOWED_PEOPLE = 4  # Pictures with more people will be ignored in training
GAUSSIAN_VAR = 1.1  # Variance for 2D gaussian

TRAIN_FOLDER = "/media/zc/Ext4-1TB/AI_challenger_keypoint"
RNN_SAVED_JOINTS_FOLDER = "./dataset/gen/rnn_saved_joints"
VIDEO_FOLDER_PATH = "dataset/policepose_video"
SUBTITLE_DELAY_FRAMES = 12
RNN_HIDDEN_UNITS = 32
PCM2JOINT_THRESHOLD = 0.1  # Threshold for converting pcms to joint coordinates. Otherwise coor = -1.
NUM_PCMs = 14
NUM_PAFs = 11 * 2
RES_IMG_WH = 512
# Training Video List
VIDEO_LIST = None
# Add all videos with name train*.mp4 as training video
if VIDEO_LIST is None:
    VIDEO_LIST = glob.glob('dataset/policepose_video/train*.mp4')
    VIDEO_LIST = [os.path.basename(p) for p in VIDEO_LIST]
    VIDEO_LIST = [p.replace('.mp4', '') for p in VIDEO_LIST]
    print(VIDEO_LIST)
'''VIDEO_LIST = [
    "train10sec1",
    "train10sec2",
    "train10sec3",
    "train10sec4",
    "train10sec5",
    "train10sec6",
    "train5sec1",
    "train5sec2",
    "train5sec3",
    "train5sec4",
    "train5sec5"]'''

bones = [[1,2], [2,3], [4,5], [5,6], [14,1], [14,4], [7,8], [8,9], [10,11], [11,12], [13,14]]
bones = np.array(bones)-1

police_dict = {
                0: "--",
                1: "STOP",
                2: "MOVE STRAIGHT",
                3: "LEFT TURN",
                4: "LEFT TURN WAITING",
                5: "RIGHT TURN",
                6: "LANG CHANGING",
                7: "SLOW DOWN",
                8: "PULL OVER"}

police_dict_chinese = {
                0: "--",
                1: "停止",
                2: "直行",
                3: "左转",
                4: "左待转",
                5: "右转",
                6: "变道",
                7: "减速",
                8: "靠边停车"}


def create_necessary_folders():
    def create(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    dirs = ["./logs", "rnn_logs/", "./dataset/gen",
            VIDEO_FOLDER_PATH, "./dataset/AI_challenger_keypoint",
            RNN_SAVED_JOINTS_FOLDER]
    [create(d) for d in dirs]
