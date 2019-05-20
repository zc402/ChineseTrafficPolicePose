import cv2
import glob
import parameters as pa
import os.path as path
import numpy as np

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

if __name__ == "__main__":
    predicted_files = glob.glob(path.join(pa.RNN_PREDICT_OUT_FOLDER, "*.csv"))

    for predicted in predicted_files:
        labels = load_label(predicted)
        name = path.splitext(path.basename(predicted))[0]
        video_file = path.join(pa.VIDEO_FOLDER_PATH, name+".mp4")

        out_path = path.join(pa.SUBTITLE_VIDEO_FOLDER, name+".avi")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1080, 1080))
        print("Path to save: " + out_path)

        cap = cv2.VideoCapture(video_file)
        frame_cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:

                label_num = labels[frame_cnt]  # A gesture number
                frame_cnt = frame_cnt + 1
                label_en = pa.police_dict[label_num]  # english word
                bar = np.zeros((1080, 840, 3), dtype=np.uint8)

                cv2.putText(bar, label_en, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)
                new_frame = np.concatenate((frame, bar), axis=1)
                cv2.imshow('Saving...', new_frame)
                cv2.waitKey(5)

                out.write(new_frame)
            else:
                break
        out.release()
        cap.release()
