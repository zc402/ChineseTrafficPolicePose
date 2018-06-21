import os
import glob

# Global Parameters
PH, PW = (512, 512)
HEAT_ZOOMING_RATE = 8
assert(PW % HEAT_ZOOMING_RATE == 0 and PH % HEAT_ZOOMING_RATE == 0)
HEAT_H, HEAT_W = PW // HEAT_ZOOMING_RATE, PH // HEAT_ZOOMING_RATE
HEAT_SIZE = (HEAT_W, HEAT_H)  # 64, 64
MAX_ALLOWED_PEOPLE = 8  # Pictures with more people will be ignored in training

RESIZED_IMG_FOLDER = "dataset/gen/ai_challenger_ratio_kept"
RNN_SAVED_JOINTS_PATH = "./dataset/gen/rnn_saved_joints"
VIDEO_FOLDER_PATH = "dataset/policepose_video"
SUBTITLE_DELAY_FRAMES = 7
RNN_HIDDEN_UNITS = 32
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

bones = [[6, 7], [7, 2], [2, 1], [1, 0], [7, 3], [3, 4], [4, 5]]

police_dict = {
                0: "--",
                1: "STOP",
                2: "PASS",
                3: "TURN LEFT",
                4: "LEFT WAIT",
                5: "TURN RIGHT",
                6: "CHANGE LANE",
                7: "SLOW DOWN",
                8: "GET OFF"}

police_dict_chinese = {
                0: "--",
                1: "停止",
                2: "通行",
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
    dirs = ["./logs", "rnn_logs/", "./dataset/gen", RESIZED_IMG_FOLDER,
            VIDEO_FOLDER_PATH, "./dataset/AI_challenger_keypoint",
            RNN_SAVED_JOINTS_PATH]
    [create(d) for d in dirs]
