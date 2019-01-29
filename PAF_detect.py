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
import glob
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
        :return: joint positions [joint, xy]
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

    def show_PCMs(self, video):
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
            elif k == ord('p'):  # Pause
                cv2.waitKey(0)
        cap.release()
        detector.release()

    def show_bone_connections(self, video):
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

class SaveFeatures:
    def save_joint_percent_values(self, video):
        detector = PAF_detect()
        frame_joints = []
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise FileNotFoundError("%s can't be opened by OpenCV" % video)
        v_size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        v_fps = int(cap.get(cv2.CAP_PROP_FPS))
        if v_fps != 15:
            raise ValueError("video %s have a frame rate of %d, not 15." % (video, v_fps))
        current_frame = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            current_frame = current_frame+1
            frame = cv2.resize(frame, (pa.PW, pa.PH))
            frame = frame.astype(np.float32)
            frame = frame / 255.
            joints_xy = detector.detect_np_pic(frame)  # [Joint, xy]
            frame_joints.append(joints_xy)
            print("Parsing frame %d of %d frames" % (current_frame, v_size))
        cap.release()
        detector.release()
        frame_joints = np.asarray(frame_joints, dtype=np.float32)
        video_name = os.path.basename(video)
        video_name, _ = os.path.splitext(video_name)
        save_path = os.path.join(pa.RNN_SAVED_JOINTS_FOLDER, video_name+".npy")
        np.save(save_path, frame_joints)
        print("Video file %s parsed and saved!" % video)

    def parse_save_mp4_files(self, folder):
        wildcard_path = os.path.join(folder, "*.mp4")
        mp4_list = glob.glob(wildcard_path)
        for mp4 in mp4_list:
            self.save_joint_percent_values(mp4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect PAFs')
    parser.add_argument("file", type=str, help="video or image file path")
    parser.add_argument("-m", help="show heatmap video", default=False, action="store_true")
    parser.add_argument("-b", help="show bone video", default=False, action="store_true")
    parser.add_argument("-s", help="save joint positions to file", default=False, action="store_true")
    parser.add_argument("-a", help="parse and save all mp4 from folder", default=False, action="store_true")
    args = parser.parse_args()
    if args.m:
        ShowResults().show_PCMs(args.file)
    elif args.b:
        ShowResults().show_bone_connections(args.file)
    elif args.s:
        print("Saving joint positions...")
        SaveFeatures().save_joint_percent_values(args.file)
    elif args.a:
        print("Saving joint positions from folder")
        if os.path.isdir(args.file):
            SaveFeatures().parse_save_mp4_files(args.file)
        else:
            raise FileNotFoundError("%s is not a folder" % args.file)
