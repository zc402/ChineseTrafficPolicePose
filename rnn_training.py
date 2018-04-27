import video_utils
import tensorflow as tf
import sys
import parameters as pa
import gpu_pipeline
import gpu_network
import rnn_network
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io
import os
import pysrt

LEARNING_RATE = 0.0004
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Train or test mode")


def build_training_ops(loss_tensor):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)

    decaying_learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, global_step, 20000, 0.8, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
    grads = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    # Add summary for every gradient
    for grad, var in grads:
        if grad is not None:
            if 'rnn' in var.op.name or 'rconv' in var.op.name:  # Only summary rnn gradients
                tf.summary.histogram(var.op.name + "/gradient", grad)

    summary_op = tf.summary.merge_all()
    return [loss_tensor, global_step,
            decaying_learning_rate, train_op, summary_op]


def print_log(loss_num, g_step_num, lr_num, itr):
    INTERVAL = 100
    log_dict = {}
    log_dict['Loss'] = loss_num
    log_dict['Step'] = g_step_num
    log_dict['Learning Rate'] = lr_num
    if itr % INTERVAL == 0:
        print(log_dict)


def test_mode(sess, btjh, btc_pred_max, state, time_step):
    # BATCH_SIZE is always 1 under test mode
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
    v_name = "test"
    # video_utils.save_joints_position(v_name)
    # TODO: Read joint pos file
    joint_data = np.load(os.path.join(pa.RNN_SAVED_JOINTS_PATH, v_name + ".npy"))
    pred_list = []
    for time in range(0, len(joint_data)-time_step, time_step):
        j_step = joint_data[time: time + time_step]
        j_step = j_step[np.newaxis,:,:,:]  # Batch size = 1
        feed_dict = {btjh: j_step}
        btc_pred_num = sess.run([btc_pred_max, state], feed_dict = feed_dict)
        pred = np.reshape(btc_pred_num, [-1])
        [pred_list.append(p) for p in pred]

    file = pysrt.SubRipFile()
    for i, item in enumerate(pred_list):
        total_ms = round((1000/15) * i)
        total_s = total_ms // 1000
        total_m = total_s // 60
        start = '00:'+str(total_m %
                          60)+':'+str(total_s %
                                      60)+':'+str(total_ms %
                                                  1000)
        end = '00:'+str(total_m %
                        60)+':'+str(total_s %
                                    60)+':'+str(total_ms %
                                                1000 + 14)
        sub = pysrt.SubRipItem(i, start=start, end=end,
                               text=police_dict[int(item)])
        file.append(sub)
    file.save(os.path.join(pa.VIDEO_FOLDER_PATH, 'test.srt'))


def main(argv=None):
    if 'test' in FLAGS.mode:
        BATCH_SIZE = 1
    else:
        BATCH_SIZE = 30
    TIME_STEP = 15
    NUM_CLASSES = 9  # 8 classes + 1 for no move
    NUM_JOINTS = 8

    # batch_time_joint_holder:
    btjh = tf.placeholder(tf.float32, [BATCH_SIZE, TIME_STEP, NUM_JOINTS, 2])  # 2:xy
    # batch_time label(classes) holder
    btlh = tf.placeholder(tf.int32, [BATCH_SIZE, TIME_STEP])
    with tf.variable_scope("rnn-net"):
        # b t c(0/1)
        btl_onehot = tf.one_hot(btlh, NUM_CLASSES, axis=-1)
        img_j_xy = tf.reshape(btjh, [-1, NUM_JOINTS, 2])
        img_fe = rnn_network.extract_features_from_joints(img_j_xy)
        btf = tf.reshape(img_fe, [BATCH_SIZE, TIME_STEP, -1])
        pred, state = rnn_network.build_rnn_network(btf, NUM_CLASSES)
        loss = rnn_network.build_rnn_loss(pred, btl_onehot)
        lgdts_tensor = build_training_ops(loss)

    sess = tf.Session()

    all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    rnn_var = [var for var in all_var]
    rnn_saver = tf.train.Saver()
    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
    else:
        sess.run(tf.variables_initializer(rnn_var))

    # summary_writer = tf.summary.FileWriter("rnn_logs/summary", sess.graph)

    # model evaluation
    btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
    btc_pred_max = tf.argmax(btc_pred, 2)
    l_max = tf.argmax(btl_onehot, 2)
    correct_prediction = tf.equal(tf.argmax(btc_pred, 2), tf.argmax(btl_onehot, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if 'test' in FLAGS.mode:
        test_mode(sess, btjh, btc_pred_max, state, TIME_STEP)
        sess.close()
        exit(0)

    # Load joint pos
    jgen = video_utils.random_btj_btl_gen(BATCH_SIZE, TIME_STEP)
    for itr in range(1, int(1e7)):
        btj, btl = next(jgen)
        feed_dict = {btjh: btj, btlh: btl}
        loss_num, g_step_num, lr_num, train_op = sess.run(
            lgdts_tensor[0:4], feed_dict=feed_dict)
        print_log(loss_num, g_step_num, lr_num, itr)

        # Summary
        if itr % 5000 == 0:
            btc_pred_num, l_max_num, acc = sess.run(
                [btc_pred_max, l_max, accuracy],
                feed_dict=feed_dict)
            print("pred:  "+str(btc_pred_num))
            print("label: "+str(l_max_num))
            print("accuracy: " + str(acc))
            # summary_str = sess.run(lgdts_tensor[4], feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, g_step_num)

            rnn_saver.save(sess, "rnn_logs/ckpt")
            print('Model Saved.')

    sess.close()


if __name__ == "__main__":
    tf.app.run()
exit(0)
