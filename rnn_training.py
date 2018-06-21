import video_utils
import tensorflow as tf
import sys
import parameters as pa
import rnn_network
import numpy as np
import os

LEARNING_RATE = 0.001
FLAGS = tf.flags.FLAGS

def build_training_ops(loss_tensor):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)

    decaying_learning_rate = tf.train.exponential_decay(
        LEARNING_RATE, global_step, 10000, 0.8, staircase=True)

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
    INTERVAL = 10
    log_dict = {}
    log_dict['Loss'] = loss_num
    log_dict['Step'] = g_step_num
    log_dict['Learning Rate'] = lr_num
    if itr % INTERVAL == 0:
        print(log_dict)


def main(argv=None):

    BATCH_SIZE = 256
    TIME_STEP = 15 * 90
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
        pred, state = rnn_network.build_rnn_network(btf, NUM_CLASSES, training=True)
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

    print("Training with batch size:"+str(BATCH_SIZE))
    # Load joint pos
    jgen = video_utils.random_btj_btl_gen(BATCH_SIZE, TIME_STEP)
    for itr in range(1, int(4e4)):
        btj, btl = next(jgen)
        feed_dict = {btjh: btj, btlh: btl}
        loss_num, g_step_num, lr_num, train_op = sess.run(
            lgdts_tensor[0:4], feed_dict=feed_dict)
        print_log(loss_num, g_step_num, lr_num, itr)

        # Summary
        if itr % 200 == 0:
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

assert sys.version_info >= (3,5)
if __name__ == "__main__":
    tf.app.run()
exit(0)
