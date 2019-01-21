"""
Use PAF detector to output human keypoint results
The results are used as dataset for RNN
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import sys
import argparse

# import gpu_pipeline
import PAF_network
import parameters as pa


class PAF_detect:
    def __init__(self):
        tf.reset_default_graph()
        pa.create_necessary_folders()
        # Place Holder, batch size is 1
        self.img_holder = tf.placeholder(tf.float32, [1, pa.PH, pa.PW, 3])
        # PAF inference network
        self.tensor_paf_pcm = PAF_network.PoseNet().inference_paf_pcm(self.img_holder)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state("logs/")
        if ckpt:
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise FileNotFoundError("No PAF check point")

    def _paf_pcm_to_normalized_coordinates(self, paf_pcm):
        """
        Convert prediction output of PAF to joint coordinates (between 0.0~1.0)
        :param pafpcm:
        :return: [joint, xy] width, height of joint coor w.r.t WHOLE IMAGE
        """
        pcm = paf_pcm[:, :, :, 14:]
        pcm = np.clip(pcm, 0., 1.)
        j_xy_precent = np.zeros((pcm.shape[3], 2), np.float32)
        for idx_joint in range(pcm.shape[3]):  # This many joints
            heat = pcm[0, :, :, idx_joint]  # first and only image from batch
            c_coor_1d = np.argmax(heat)  # Index of biggest confidence value
            c_coor_2d = np.unravel_index(
                c_coor_1d, [pa.HEAT_SIZE[1], pa.HEAT_SIZE[0]])
            c_value = heat[c_coor_2d]
            # x,y coordinate of joints. [joints, coor]
            if c_value > pa.PCM2JOINT_THRESHOLD:
                percent_h = c_coor_2d[0] / pa.HEAT_H
                percent_w = c_coor_2d[1] / pa.HEAT_W

                j_xy_precent[idx_joint, 0] = percent_w
                j_xy_precent[idx_joint, 1] = percent_h
            else:
                j_xy_precent[idx_joint, :] = -1.
        return j_xy_precent

    def detect_np_pic(self, np_pic):
        """
        Detect 1 numpy picture, return normalized joint coor
        :param np_pic:
        :return: joint positions
        """
        np_pic = np_pic[np.newaxis]
        feed_dict = {self.img_holder: np_pic}
        paf_pcm = self.sess.run(self.tensor_paf_pcm, feed_dict=feed_dict)
        percent_joints = self._paf_pcm_to_normalized_coordinates(paf_pcm)
        return percent_joints

    def detect_np_pic_ret_PCMs(self, np_pic):
        """
        Detect 1 np picture, return Heatmaps
        :param np_pic:
        :return: Heatmaps [batch, h, w, joints]
        """
        np_pic = np_pic[np.newaxis]
        feed_dict = {self.img_holder: np_pic}
        paf_pcm = self.sess.run(self.tensor_paf_pcm, feed_dict=feed_dict)
        pcm = paf_pcm[:, :, :, 14:]
        pcm = np.clip(pcm, 0., 1.)
        return pcm

    def release(self):
        self.sess.close()

class ShowResults:
    def video_to_heatmaps(self, video):
        detector = PAF_detect()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened by OpenCV" % video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frame = frame.astype(np.float32)
            frame = frame / 255.
            heatmaps = detector.detect_np_pic_ret_PCMs(frame)
            heatmaps = heatmaps.max(axis=(0,3))
            # Enlarge heat image
            heatmaps = cv2.resize(heatmaps, (512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("frame", frame)
            cv2.imshow("heatmaps", heatmaps)
            k = cv2.waitKey(50)
            if k == 27:  # ESC
                break
        cap.release()
        detector.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect PAFs')
    parser.add_argument("file", type=str, help="video or image file path")
    parser.add_argument("-m", help="show video heatmaps", default=False, action="store_true")
    args = parser.parse_args()
    if args.m:
        ShowResults().video_to_heatmaps(args.file)