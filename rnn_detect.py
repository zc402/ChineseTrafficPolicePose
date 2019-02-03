import video_utils
import tensorflow as tf
import sys
import parameters as pa
import rnn_network
import argparse
import numpy as np
import os

def infer_npy(npy_path):
    tjc = np.load(npy_path)
    btjc = tjc[np.newaxis]

    BATCH_SIZE = 1
    TIME_STEP = tjc.shape[0]
    NUM_GESTURE_CLASSES = 9  # Not used

    # batch_time_feature holder:
    tf_btf = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP, 30])  # 10 length + 20 angle
    # batch_time label(classes) holder
    tf_btl = tf.placeholder(tf.int32, [BATCH_SIZE, TIME_STEP])
    with tf.variable_scope("rnn-net"):
        # b t c(0/1)
        btl_onehot = tf.one_hot(tf_btl, NUM_GESTURE_CLASSES, axis=-1)
        pred, state = rnn_network.build_rnn_network(tf_btf, NUM_GESTURE_CLASSES, training=False)

    sess = tf.Session()

    rnn_saver = tf.train.Saver()
    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
    else:
        raise RuntimeError("No check point save file.")

    # summary_writer = tf.summary.FileWriter("rnn_logs/summary", sess.graph)

    # model evaluation
    btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
    bt_pred = tf.argmax(btc_pred, 2)
    l_max = tf.argmax(btl_onehot, 2)
    correct_prediction = tf.equal(tf.argmax(btc_pred, 2), tf.argmax(btl_onehot, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Training with batch size:" + str(BATCH_SIZE))
    # Load joint pos

    # Build network

    btf = rnn_network.extract_bone_length_joint_angle(btjc)
    feed_dict = {tf_btf: btf}

    bt_pred_num, l_max_num, acc = sess.run(
        [bt_pred, l_max, accuracy],
        feed_dict=feed_dict)
    print("pred:  " + str(bt_pred_num))
    print("label: " + str(l_max_num))
    print("accuracy: " + str(acc))

    sess.close()

    def save_label(label_list, csv_file):
        """

        :param label_list: a list of int
        :param csv_file:
        :return:
        """
        str_list = ["%d" % e for e in label_list]
        str_line = ",".join(str_list)

        with open(csv_file, 'w') as label_file:
            label_file.write(str_line)
        print("saved: %s" % csv_file)

    save_name = os.path.basename(npy_path).replace(".npy", ".csv")
    save_path = os.path.join(pa.RNN_PREDICT_OUT_FOLDER, save_name)
    save_label(bt_pred_num[0], save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detect gestures')
    parser.add_argument("file", type=str, help="video path")
    args = parser.parse_args()
    infer_npy(args.file)
