# from skimage.viewer import ImageViewer
import tensorflow as tf
import sys
import parameters as pa
import PAF_network
import numpy as np
import os
from skimage.draw import line_aa


assert sys.version_info >= (3, 5)


def _class_per_frame(srt, total_frames, frame_rate=15):
    """
    Convert srt subtitle to class per frame
    :param srt: subtitle path
    :param delay: delay the appearance of prediction
    :return: class per frame array
    """
    delay = pa.SUBTITLE_DELAY_FRAMES
    subs = pysrt.open(srt)
    # Time of each frame (Millisecond)
    time_of_frame_list = [time / frame_rate * 1000
                          for time in range(total_frames)]

    def class_of_one_frame(frame_num):
        """
        :param frame_num:
        :return: class of designated frame
        """
        for sub in subs:
            if sub.start.ordinal < time_of_frame_list[frame_num] < sub.end.ordinal:
                # Check if subtilte in range 1~8
                text = sub.text_without_tags
                assert 1 <= int(text) <=8, "wrong tag in subtilte file"
                return text
        # No subtitle annotated
        return "0"
    class_list = [class_of_one_frame(num) for num in range(total_frames)]
    delay_list = ["0" for i in range(delay)]
    class_list = delay_list + class_list
    return class_list





def skeleton_video(name):
    """
    Generate skeleton video from joints position file
    :return:
    """
    map_h = 512
    map_w = 512
    joint_data_path = os.path.join(
        pa.RNN_SAVED_JOINTS_PATH,
        name + ".npy")
    joint_data = np.load(joint_data_path)
    video = []
    for joint_xy in joint_data:
        # Inside one image
        frame = np.zeros([map_h, map_w], dtype=np.uint8)
        for b1, b2 in pa.bones:
            if np.less(
                    joint_xy[b1, :],
                    0).any() or np.less(
                    joint_xy[b2, :],
                    0).any():
                continue  # no detection
            x1 = int(joint_xy[b1, 0] * map_w)
            y1 = int(joint_xy[b1, 1] * map_h)
            x2 = int(joint_xy[b2, 0] * map_w)
            y2 = int(joint_xy[b2, 1] * map_h)
            rr, cc, val = line_aa(y1, x1, y2, x2)
            frame[rr, cc] = 255

        video.append(frame)
    skvideo.io.vwrite(
        "skeleton.mp4",
        video,
        inputdict={
            '-r': '15/1'},
        outputdict={
             '-r': '15/1'})


def random_btj_btl_gen(batch_size, time_steps):
    "Load joint pos with labels at random time"
    video_names = pa.VIDEO_LIST
    joints_paths = [os.path.join(pa.RNN_SAVED_JOINTS_PATH, video_name + ".npy") for video_name in video_names]
    # Joints: [F] I J XY. Joints in each film
    joints_data_list = [np.load(p) for p in joints_paths]
    fe_length_list = [len(j) for j in joints_data_list]
    # srt subtitle file path
    srt_path_list = [os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".srt") for video_name in video_names]
    # labels for each film
    i_c_labels_list  = [_class_per_frame(srt_path, fe_length, 15) for srt_path, fe_length in list(zip(srt_path_list, fe_length_list))]

    while True:
        # Batch_time_joints: B T J
        batch_time_joints_list = []
        # Batch_time_labels: B T
        batch_labels_list = []
        for batch in range(0, batch_size):
            # Output 1 batch
            film_ind = np.random.randint(0, len(i_c_labels_list))
            labels = i_c_labels_list[film_ind]
            fe_length = fe_length_list[film_ind]
            # Start index of time
            start = np.random.randint(0, fe_length - time_steps)
            joints = joints_data_list[film_ind]
            time_joints = joints[start: start + time_steps]
            time_labels = labels[start: start + time_steps]
            batch_time_joints_list.append(time_joints)
            batch_labels_list.append(time_labels)
        yield (batch_time_joints_list, batch_labels_list)


def video_frame_class_gen(batch_size, time_steps):
    """
    Generate frames with corresponding labels
    :param batch_size: RNN batch size
    :param time_steps: Consecutive frames for 1 batch
    :return: batch_time_frames, batch_time_labels
    """
    video_name = pa.VIDEO_LIST[0]
    video_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".mp4")
    srt_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".srt")
    frames = skvideo.io.vread(video_path)[
        900:990]  # TODO: This is just for test!
    frames_resize = []
    for num, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=np.float32)
        # frame = resize_keep_ratio(frame, (frame.shape[1], frame.shape[0]), (pa.PW, pa.PH))
        frame = frame / 255.
        frames_resize.append(frame)
    frames = None

    num_frames = len(frames_resize)
    labels = _class_per_frame(srt_path, num_frames, 15)  # Frame rate 15!
    while True:
        start_idx_list = np.random.randint(
            0, num_frames - time_steps, size=batch_size)
        # [B,T,H,W,C]
        batch_time_frames = [frames_resize[start: start + time_steps]
                             for start in start_idx_list]
        # [B,T]
        batch_time_labels = [labels[start: start + time_steps]
                             for start in start_idx_list]
        yield (batch_time_frames, batch_time_labels)

def save_all_training_samples_to_joint_data():
    [save_joints_position(vname) for vname in pa.VIDEO_LIST]
