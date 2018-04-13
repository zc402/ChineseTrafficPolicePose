import skvideo.io
from skimage.viewer import ImageViewer
import pysrt
import tensorflow as tf
import sys
import parameters
import gpu_pipeline
import gpu_network
import numpy as np
import os
from PIL import Image

assert sys.version_info >= (3, 5)

# video = skvideo.io.vread("dataset/policepose_video/20180412.mp4")#,num_frames=4*1800)
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

def class_per_frame(srt, total_frames):
    """
    Convert srt subtitle to class per frame array assuming constant frame rate of 30
    :param srt: subtitle path
    :return: class per frame array
    """
    subs = pysrt.open(srt)
    # Time of each frame (Millisecond)
    time_of_frame_list = [time / 30 * 1000 for time in range(total_frames)]
    
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
    
    frame_class_list = [[num, class_of_one_frame(num)] for num in range(total_frames)]
    return frame_class_list


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

def save_evaluated_heatmaps():
    parameters.create_necessary_folders()
    # Place Holder
    PH, PW = parameters.PH, parameters.PW
    img_holder = tf.placeholder(tf.float32, [1, PH, PW, 3])
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

                save_folder_path = os.path.join(parameters.RNN_SAVED_HEATMAP_PATH, folder_name)
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)
                save_path = os.path.join(save_folder_path, str(frame_num) + ".npy")
                np.save(save_path, paf_pcm)
                print(save_path)
        
        # Load frame from video
        for video_name in parameters.VIDEO_LIST:
            video_path = os.path.join(parameters.VIDEO_FOLDER_PATH, video_name + ".m4v")
            video_gen = skvideo.io.vreader(video_path)
            
            for num, frame in enumerate(video_gen):
                frame = np.asarray(frame, dtype=np.float32)
                frame = resize_keep_ratio(frame,(frame.shape[1], frame.shape[0]) ,(parameters.PW, parameters.PH))
                frame = frame / 255.
                frame = frame[np.newaxis, :, :, :]
                save_frame_paf_pcm_to_file(frame, video_name, num)
                
save_evaluated_heatmaps()
    
