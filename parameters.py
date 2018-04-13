import os

# Global Parameters
PH, PW = (512, 512)
HEAT_SIZE = (PW/8, PH/8) # 64, 64

RNN_SAVED_HEATMAP_PATH = "./dataset/gen/rnn_saved_heatmap"


def create_necessary_folders():
    def create(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    dirs = ["./logs", "./dataset/gen",
            "./dataset/policepose_video", "./dataset/AI_challenger_keypoint"]
    map(create, dirs)