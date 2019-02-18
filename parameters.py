"""
Global parameters
"""
import os
import glob
import numpy as np

PH, PW = (512, 512)  # Size of network input picture
HEAT_ZOOMING_RATE = 8  # Final output heatmap is zoomed by this factor during network processing.
assert(PW % HEAT_ZOOMING_RATE == 0 and PH % HEAT_ZOOMING_RATE == 0)
HEAT_H, HEAT_W = PW // HEAT_ZOOMING_RATE, PH // HEAT_ZOOMING_RATE  # Size of output heatmap
HEAT_SIZE = (HEAT_W, HEAT_H)  # A tuple of heatmap size
MAX_ALLOWED_PEOPLE = 4  # Pictures with more people will be ignored in training
GAUSSIAN_VAR = 1.1  # Variance for 2D gaussian

TRAIN_FOLDER = "/media/zc/Ext4-1TB/AI_challenger_keypoint"
RNN_SAVED_JOINTS_FOLDER = "./dataset/gen/rnn_saved_joints"
VIDEO_FOLDER_PATH = "dataset/policepose_video"
# RNN_TRAIN_FOLDER = "dataset/rnn_train_videos"
LABEL_CSV_FOLDER_TRAIN = "dataset/csv_train"  # Training label folder
LABEL_CSV_FOLDER_TEST = "dataset/csv_test"  # Training label folder
RNN_PREDICT_OUT_FOLDER = "dataset/rnn_out"  # Predicted labels folder
RNN_HIDDEN_UNITS = 32  # Number of dense units in RNN
PCM2JOINT_THRESHOLD = 0.1  # Threshold for converting pcms to joint coordinates. Otherwise coor = -1.
NUM_PCMs = 14  # Number of PCMs
NUM_PAFs = 11 * 2  # Number of PAFs
RES_IMG_WH = 512  # PAF_detect output image's width and height
LABEL_DELAY_FRAMES = 15  # This many frames are delayed to leave some time for RNN to observe the gesture
# Add all videos with name train*.mp4 as training video

bones = [[1,2], [2,3], [4,5], [5,6], [14,1], [14,4], [7,8], [8,9], [10,11], [11,12], [13,14]]  # Bone connections.
bones = np.array(bones)-1  # Start from index 0
bones_body = bones[:10]
bones_head = bones[10:]
assert bones_head.shape[0] == 1

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
    dirs = ["./logs", "rnn_logs/", "./dataset/gen", LABEL_CSV_FOLDER_TRAIN, RNN_PREDICT_OUT_FOLDER,LABEL_CSV_FOLDER_TEST,
            VIDEO_FOLDER_PATH,
            RNN_SAVED_JOINTS_FOLDER]
    [create(d) for d in dirs]

if __name__ == "__main__":
    create_necessary_folders()