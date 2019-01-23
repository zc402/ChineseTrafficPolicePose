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
        pcm = paf_pcm[:, :, :, pa.NUM_PAFs:]
        pcm = np.clip(pcm, 0., 1.)
        j_xy_precent = np.zeros((pa.NUM_PCMs, 2), np.float32)
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
            else:  # Not detected
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
        pcm = paf_pcm[:, :, :, pa.NUM_PAFs:]
        pcm = np.clip(pcm, 0., 1.)
        return pcm

    def release(self):
        self.sess.close()

    def save_joint_positions(self, video_path):
        """
        Predict joint positions. (percent)
        Save joint positions from a video to file
        :param video_path: path of video
        :return:
        """
        # video_path = os.path.join(pa.VIDEO_FOLDER_PATH, v_name + ".mp4")
        detector = PAF_detect()
        cap = cv2.VideoCapture(video_path)
        frame_jxy = []  # shape [frame, joint, xy]
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened by OpenCV" % video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (pa.HEAT_W, pa.HEAT_H), interpolation=cv2.INTER_CUBIC)
            frame = frame.astype(np.float32)
            frame = frame / 255.
            percent_joints = detector.detect_np_pic(frame)
            frame_jxy.append(percent_joints)

        video_name = os.path.basename(video_path)
        video_name, ext = os.path.splitext(video_name)
        save_path = os.path.join(
            pa.RNN_SAVED_JOINTS_PATH,
            video_name + ".npy")
        if os.path.exists(save_path):
            print("Override old file")
            os.remove(save_path)
        frame_jxy = np.asarray(frame_jxy)
        np.save(save_path, frame_jxy)
        print("Joints saved: %s " % save_path)

class ShowResults:
    def __init__(self):
        gray14 = np.linspace(0, 255, 14, endpoint=False, dtype=np.uint8)
        gray14.reshape([1, 14])  # h,w for gray image
        self.color14 = cv2.applyColorMap(gray14, cv2.COLORMAP_RAINBOW)
        self.color14 = self.color14.reshape([14, 3])

    """
    def _assign_color(self, hms):

        imgs = []
        for i in range(14):
            hm = hms[:,:, i]
            img3 = hm[:,:, np.newaxis] * self.color14[i]
            imgs.append(img3)
        # imgs: [img,h,w,c]
        # colored_img = np.sum(imgs, axis=0)
        colored_img = imgs[0]
        colored_img = colored_img.astype(np.uint8)
        colored_img = np.clip(colored_img, 0, 255)
        return colored_img
    """

    def video_to_heatmaps(self, video):
        detector = PAF_detect()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened by OpenCV" % video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (pa.PW, pa.PH))
            frame = frame.astype(np.float32)
            frame = frame / 255.
            hms = detector.detect_np_pic_ret_PCMs(frame)  # Heatmaps: 0~1

            hms = hms.max(axis=(0,3))
            hms = cv2.resize(hms, (512, 512), interpolation=cv2.INTER_CUBIC)
            # Enlarge heat image
            cv2.imshow("frame", frame)
            cv2.imshow("heatmaps", hms)
            k = cv2.waitKey(5)
            if k == 27:  # ESC
                break
        cap.release()
        detector.release()

    def video_to_bones(self, video):
        detector = PAF_detect()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened by OpenCV" % video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (pa.PW, pa.PH))
            frame = frame.astype(np.float32)
            frame = frame / 255.
            percent_joints = detector.detect_np_pic(frame)

            # Draw bones
            img_bones = np.zeros([pa.RES_IMG_WH, pa.RES_IMG_WH, 3], dtype=np.uint8)
            for b_num, (b1, b2) in enumerate(pa.bones):
                if np.less(percent_joints[b1, :], 0).any() or \
                        np.less(percent_joints[b2, :], 0).any():
                    continue  # no detection
                x1 = int(percent_joints[b1, 0] * pa.RES_IMG_WH)
                y1 = int(percent_joints[b1, 1] * pa.RES_IMG_WH)
                x2 = int(percent_joints[b2, 0] * pa.RES_IMG_WH)
                y2 = int(percent_joints[b2, 1] * pa.RES_IMG_WH)
                color = self.color14[b_num].tolist()
                cv2.line(img_bones, (x1, y1), (x2, y2), color, 4)

            # Enlarge heat image
            cv2.imshow("frame", frame)
            cv2.imshow("bones", img_bones)
            k = cv2.waitKey(5)
            if k == 27:  # ESC
                break
            elif k == ord('p'):
                cv2.waitKey(0)
        cap.release()
        detector.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect PAFs')
    parser.add_argument("file", type=str, help="video or image file path")
    parser.add_argument("-m", help="show heatmap video", default=False, action="store_true")
    parser.add_argument("-b", help="show bone video", default=False, action="store_true")
    parser.add_argument("-s", help="save joint positions to file", default=False, action="store_true")
    args = parser.parse_args()
    if args.m:
        ShowResults().video_to_heatmaps(args.file)
    elif args.b:
        ShowResults().video_to_bones(args.file)
    elif args.s:
        print("Saving joint positions...")
        PAF_detect().save_joint_positions(args.file)
