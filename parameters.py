import os

# Global Parameters
PH, PW = (512, 512)
HEAT_ZOOMING_RATE = 8
HEAT_SIZE = (PW/HEAT_ZOOMING_RATE, PH/HEAT_ZOOMING_RATE) # 64, 64
MAX_ALLOWED_PEOPLE = 8

RNN_SAVED_HEATMAP_PATH = "./dataset/gen/rnn_saved_heatmap"
VIDEO_FOLDER_PATH = "dataset/policepose_video"
VIDEO_LIST = ["20180412"]


def create_necessary_folders():
    def create(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    dirs = ["./logs", "./dataset/gen",
            VIDEO_FOLDER_PATH, "./dataset/AI_challenger_keypoint",
            RNN_SAVED_HEATMAP_PATH]
    map(create, dirs)