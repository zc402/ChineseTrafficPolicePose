import os

# Global Parameters
PH, PW = (512, 512)
HEAT_ZOOMING_RATE = 8
assert(PW%HEAT_ZOOMING_RATE == 0 and PH%HEAT_ZOOMING_RATE == 0)
HEAT_H, HEAT_W = PW // HEAT_ZOOMING_RATE ,PH // HEAT_ZOOMING_RATE
HEAT_SIZE = (HEAT_W, HEAT_H) # 64, 64
MAX_ALLOWED_PEOPLE = 8

RESIZED_IMG_FOLDER = "dataset/gen/ai_challenger_ratio_kept"
RNN_SAVED_JOINTS_PATH = "./dataset/gen/rnn_saved_joints"
VIDEO_FOLDER_PATH = "dataset/policepose_video"
VIDEO_LIST = ["20180412"]

bones = [[6, 7], [7, 2], [2, 1], [1, 0], [7, 3], [3, 4], [4, 5]]

def create_necessary_folders():
    def create(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    dirs = ["./logs", "rnn_logs/", "./dataset/gen", RESIZED_IMG_FOLDER,
            VIDEO_FOLDER_PATH, "./dataset/AI_challenger_keypoint",
            RNN_SAVED_JOINTS_PATH]
    [create(d) for d in dirs]
