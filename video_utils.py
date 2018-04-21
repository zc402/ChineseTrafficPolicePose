import skvideo.io
from skimage.viewer import ImageViewer
import pysrt
import tensorflow as tf
import sys
import parameters as pa
import gpu_pipeline
import gpu_network
import numpy as np
import os
from PIL import Image
from skimage.draw import line_aa


assert sys.version_info >= (3, 5)

def _class_per_frame(srt, total_frames, frame_rate):
    """
    Convert srt subtitle to class per frame
    :param srt: subtitle path
    :return: class per frame array
    """
    subs = pysrt.open(srt)
    # Time of each frame (Millisecond)
    time_of_frame_list = [time / frame_rate * 1000 for time in range(total_frames)]
    
    def class_of_one_frame(frame_num):
        """
        :param frame_num:
        :return: class of designated frame
        """
        for sub in subs:
            if sub.start.ordinal < time_of_frame_list[frame_num] < sub.end.ordinal:
                return sub.text_without_tags
        # No subtitle annotated
        return "0"
    
    class_list = [class_of_one_frame(num) for num in range(total_frames)]
    return class_list

# Deprecated
def video_frame_generator(video_path):
    videogen = skvideo.io.vreader(video_path)
    for frame in videogen:
        viewer = ImageViewer(frame)
        viewer.show()

def resize_keep_ratio(img, ori_size, new_size):
    assert len(ori_size) == 2
    assert len(new_size) == 2
    PW, PH = new_size
    target_ratio = PW / PH
    ori_ratio = ori_size[0] / ori_size[1]
    bg = None
    if ori_ratio >= target_ratio:
        # Depends on width
        zoom_ratio = PW / ori_size[0]
        bg = np.zeros((int(ori_size[0] / target_ratio), ori_size[0], 3), np.uint8)

    elif ori_ratio < target_ratio:
        # Depends on height
        zoom_ratio = PH / ori_size[1]
        bg = np.zeros((ori_size[1], int(ori_size[1] * target_ratio), 3), np.uint8)
        
    bg[:ori_size[1], :ori_size[0], :] = bg[:ori_size[1], :ori_size[0], :] + img[:, :, :]
    re_im = Image.fromarray(bg, 'RGB')
    re_im = re_im.resize((PW, PH), Image.ANTIALIAS)
    return np.asarray(re_im)

def save_joints_position():
    """
    Save joints position from a video to file
    :return:
    """
    pa.create_necessary_folders()
    batch_size = 10
    video_path = os.path.join(pa.VIDEO_FOLDER_PATH, pa.VIDEO_LIST[0] + ".mp4")
    metadata = skvideo.io.ffprobe(video_path)
    total_frames = int(metadata["video"]["@nb_frames"])

    v_width = int(metadata["video"]["@width"])
    v_height = int(metadata["video"]["@height"])
    assert(v_height == pa.PH and v_width == pa.PW)
    v_gen = skvideo.io.vreader(video_path)

    # Place Holder
    img_holder = tf.placeholder(tf.float32, [batch_size, v_height, v_width, 3])
    # Entire network
    paf_pcm_tensor = gpu_network.PoseNet().inference_paf_pcm(img_holder)
    
    # Place for argmax values
    joint_ixy = list() # [i][j1~6][x,y]
    # Session Saver summary_writer
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("logs/")
        if ckpt:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("Tensorflow ckpt not found")
        
        # Close the graph so no op can be added
        tf.get_default_graph().finalize()

        for i in range(0, total_frames - batch_size + 1, batch_size):
            frames = [next(v_gen)/255. for _ in range(batch_size)]
            feed_dict = {img_holder: frames}
            paf_pcm = sess.run(paf_pcm_tensor, feed_dict=feed_dict)
            pcm = paf_pcm[:,:,:,10:]
            pcm = np.clip(pcm, 0., 1.)
            for idx_img in range(batch_size):
                # 6 joint in image
                img_j6 = []
                for idx_joint in range(6):
                    heat = pcm[idx_img,:,:,idx_joint]
                    c_coor_1d = np.argmax(heat)
                    c_coor_2d = np.unravel_index(c_coor_1d, [pa.HEAT_SIZE[1], pa.HEAT_SIZE[0]])
                    c_value = heat[c_coor_2d]
                    j_xy = [] # x,y
                    if c_value > 0.3:
                        percent_h = c_coor_2d[0] / pa.HEAT_H
                        percent_w = c_coor_2d[1] / pa.HEAT_W
                        j_xy.append(percent_w)
                        j_xy.append(percent_h)
                    else:
                        j_xy.append(-1.)
                        j_xy.append(-1.)
                    img_j6.append(j_xy)
                joint_ixy.append(img_j6)
            print("Image: "+str(i))
    # sess closed
    save_path = os.path.join(pa.RNN_SAVED_JOINTS_PATH, pa.VIDEO_LIST[0] + ".npy")
    np.save(save_path, joint_ixy)
    print(save_path)

def skeleton_video():
    map_h = 512
    map_w = 512
    joint_data_path = os.path.join(pa.RNN_SAVED_JOINTS_PATH, pa.VIDEO_LIST[0] + ".npy")
    joint_data = np.load(joint_data_path)
    video = []
    for joint_xy in joint_data:
        # Inside one image
        frame = np.zeros([map_h, map_w], dtype=np.uint8)
        for i in range(5): # 01234
            if np.less(joint_xy[i:i+2, :], 0).any(): continue # no detection
            x1 = int(joint_xy[i, 0] * map_w)
            y1 = int(joint_xy[i, 1] * map_h)
            x2 = int(joint_xy[i+1, 0] * map_w)
            y2 = int(joint_xy[i+1, 1] * map_h)
            rr, cc, val = line_aa(y1,x1,y2,x2)
            frame[rr, cc] = 255
        video.append(frame)
    skvideo.io.vwrite("skeleton.mp4", video, inputdict={'-r': '15/1'}, outputdict={
      '-r': '15/1'})
            
            
        
# Deprecated
def load_evaluated_heatmaps(batch_size):
    for video_name in pa.VIDEO_LIST:
        video_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".m4v")
        paf_path = os.path.join(pa.RNN_SAVED_JOINTS_PATH, video_name)
        srt_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".srt")

        metadata = skvideo.io.ffprobe(video_path)
        total_frame_num = metadata["video"]["@nb_frames"]
    
def video_frame_class_gen(batch_size, time_steps):
    """
    Generate frames with corresponding labels
    :param batch_size: RNN batch size
    :param time_steps: Consecutive frames for 1 batch
    :return:
    """
    video_name = pa.VIDEO_LIST[0]
    video_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".mp4")
    srt_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".srt")
    frames = skvideo.io.vread(video_path)[900:990] #TODO: This is just for test!
    frames_resize = []
    for num, frame in enumerate(frames):
        frame = np.asarray(frame, dtype=np.float32)
        # frame = resize_keep_ratio(frame, (frame.shape[1], frame.shape[0]), (pa.PW, pa.PH))
        frame = frame / 255.
        frames_resize.append(frame)
    frames = None
    
    num_frames = len(frames_resize)
    labels = _class_per_frame(srt_path, num_frames, 15)# Frame rate 15!
    while True:
        start_idx_list = np.random.randint(0, num_frames - time_steps, size=batch_size)
        # [B,T,H,W,C]
        batch_time_frames = [frames_resize[start : start + time_steps] for start in start_idx_list]
        # [B,T]
        batch_time_labels = [labels[start : start + time_steps] for start in start_idx_list]
        yield (batch_time_frames, batch_time_labels)

def test_video_frames(num_img):
    vgen = skvideo.io.vreader(os.path.join(pa.VIDEO_FOLDER_PATH, "test.mp4"))
    yield [vgen.next() in range(num_img)]

