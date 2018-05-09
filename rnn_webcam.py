import parameters as pa
import tensorflow as tf
import gpu_network
import rnn_network
import cv2
import numpy as np

def main(argv=None):
    g_1 = tf.Graph()
    g_2 = tf.Graph()
    with g_1.as_default():
        batch_size = 1

        # Place Holder
        img_holder = tf.placeholder(tf.float32, [batch_size, pa.PH, pa.PW, 3])
        # Entire network
        paf_pcm_tensor = gpu_network.PoseNet().inference_paf_pcm(img_holder)

    with g_2.as_default():
        NUM_CLASSES = 9  # 8 classes + 1 for no move
        NUM_JOINTS = 8

        # batch_time_joint_holder:
        btjh = tf.placeholder(tf.float32, [1, 1, NUM_JOINTS, 2])  # 2:xy
        with tf.variable_scope("rnn-net"):
            # b t c(0/1)
            img_j_xy = tf.reshape(btjh, [-1, NUM_JOINTS, 2])
            img_fe = rnn_network.extract_features_from_joints(img_j_xy)
            btf = tf.reshape(img_fe, [1, 1, -1])
            pred, state = rnn_network.build_rnn_network(btf, NUM_CLASSES)
            # model evaluation
            btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
            btc_pred_max = tf.argmax(btc_pred, 2)


    # Session Saver
    sess1 = tf.Session(graph=g_1)
    sess2 = tf.Session(graph=g_2)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state("logs/")
    if ckpt:
        saver.restore(sess1, ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("CPM ckpt not found.")

    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        saver.restore(sess2, rnn_ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("RNN ckpt not found.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH,512);
    cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT,512);

    while(True):
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('PolicePose', frame)
        rgb_norm = rgb / 255.

        feed_dict = {img_holder: rgb_norm}
        paf_pcm = sess1.run(paf_pcm_tensor, feed_dict=feed_dict)
        pcm = paf_pcm[:, :, :, 14:]
        pcm = np.clip(pcm, 0., 1.)

        # 6 joint in image
        img_j6 = []
        for idx_joint in range(8):
            heat = pcm[1, :, :, idx_joint]
            c_coor_1d = np.argmax(heat)
            c_coor_2d = np.unravel_index(
                c_coor_1d, [pa.HEAT_SIZE[1], pa.HEAT_SIZE[0]])
            c_value = heat[c_coor_2d]
            j_xy = []  # x,y
            if c_value > 0.15:
                percent_h = c_coor_2d[0] / pa.HEAT_H
                percent_w = c_coor_2d[1] / pa.HEAT_W
                j_xy.append(percent_w)
                j_xy.append(percent_h)
            else:
                j_xy.append(-1.)
                j_xy.append(-1.)
            # Joint points
            img_j6.append(j_xy)  # [j][XY]

        police_dict = {
                0: "--",
                1: "STOP",
                2: "PASS",
                3: "TURN LEFT",
                4: "LEFT WAIT",
                5: "TURN RIGHT",
                6: "CNG LANE",
                7: "SLOW DOWN",
                8: "GET OFF"}
        # video_utils.save_joints_position(v_name)
        joint_data = img_j6[np.newaxis, np.newaxis, :, :]
        feed_dict = {btjh: joint_data}
        btc_pred_num = sess2.run(btc_pred_max, feed_dict=feed_dict)
        pred = np.reshape(btc_pred_num, [-1])
        assert(len(pred) == 0)
        print(police_dict(pred[0]))

    cap.release()
    cv2.destroyAllWindows()

    sess1.close()
    sess2.close()

if __name__ == "__main__":
    tf.app.run()

