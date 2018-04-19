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

assert sys.version_info >= (3, 5)

def test():
    videogen = skvideo.io.vreader("dataset/policepose_video/20180412.m4v")
    for frame in videogen:
        viewer = ImageViewer(frame)
        viewer.show()

# print(video.shape)

# metadata = skvideo.io.ffprobe("dataset/policepose_video/20180412.m4v")
# print(metadata.keys())
# print(json.dumps(metadata["video"], indent=4))
# total_frames = metadata["video"]["@nb_frames"]

# viewer = ImageViewer(video[-1])
# viewer.show()

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

# Deprecated
def save_evaluated_heatmaps():
    pa.create_necessary_folders()
    batch_size = 10
    # Place Holder
    PH, PW = pa.PH, pa.PW
    img_holder = tf.placeholder(tf.float32, [10, PH, PW, 3])
    # Entire network
    paf_pcm_tensor = gpu_network.PoseNet().inference_paf_pcm(img_holder)
    
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
        
        # Save a single frame to np array
        def save_frame_paf_pcm_to_file(frame, folder_name, frame_num):
                feed_dict = {img_holder: frame}
                paf_pcm = sess.run(paf_pcm_tensor, feed_dict=feed_dict)

                save_folder_path = os.path.join(pa.RNN_SAVED_HEATMAP_PATH, folder_name)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                save_path = os.path.join(save_folder_path, str(frame_num) + ".npy")
                np.save(save_path, paf_pcm)
                print(save_path)
        
        # Load frame from video
        for video_name in pa.VIDEO_LIST:
            video_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".m4v")
            video_gen = skvideo.io.vreader(video_path)
            
            for num, frame in enumerate(video_gen):
                frame = np.asarray(frame, dtype=np.float32)
                frame = resize_keep_ratio(frame,(frame.shape[1], frame.shape[0]) ,(pa.PW, pa.PH))
                frame = frame / 255.
                frame = frame[np.newaxis, :, :, :]
                save_frame_paf_pcm_to_file(frame, video_name, num)

# Deprecated
def load_evaluated_heatmaps(batch_size):
    for video_name in pa.VIDEO_LIST:
        video_path = os.path.join(pa.VIDEO_FOLDER_PATH, video_name + ".m4v")
        paf_path = os.path.join(pa.RNN_SAVED_HEATMAP_PATH, video_name)
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