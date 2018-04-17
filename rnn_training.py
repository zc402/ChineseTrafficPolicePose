import video_utils
import tensorflow as tf
import sys
import parameters as pa
import gpu_pipeline
import gpu_network
import numpy as np

LEARNING_RATE = 0.0004

def build_training_ops(loss_tensor):
    """
    Build training ops
    :param loss_tensor:
    :return: [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]
    """
    global_step = tf.Variable(0, trainable=False)
    
    decaying_learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step,
                                                        20000, 0.8, staircase=True)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=decaying_learning_rate)
    grads = optimizer.compute_gradients(loss_tensor)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    
    # Add summary for every gradient
    for grad, var in grads:
        if grad is not None:
            if 'rnn' in var.op.name or 'rconv' in var.op.name:  # Only summary rnn gradients
                tf.summary.histogram(var.op.name + "/gradient", grad)
    
    summary_op = tf.summary.merge_all()
    return [loss_tensor, global_step, decaying_learning_rate, train_op, summary_op]

def print_log(loss_num, g_step_num, lr_num, itr):
    INTERVAL = 1
    log_dict = {}
    log_dict['Loss'] = loss_num
    log_dict['Step'] = g_step_num
    log_dict['Learning Rate'] = lr_num
    if itr % INTERVAL == 0:
        print(log_dict)
    
def main(argv=None):
    TIME_STEP = 15
    RNN_BATCH_SIZE = 2
    NUM_CLASSES = 9
    BUFFER = TIME_STEP * RNN_BATCH_SIZE

    img_holder = tf.placeholder(tf.float32, [TIME_STEP, pa.PH, pa.PW, 3])
    label_holder = tf.placeholder(tf.int32, [1, TIME_STEP])
    label_onehot = tf.one_hot(label_holder, NUM_CLASSES, axis=-1)
    
    poseNet = gpu_network.PoseNet()
    poseNet.set_var_trainable(False)
    poseNet.inference_paf_pcm(img_holder)
    # Convolutional Pose Machine ended
    cpm_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    poseNet.set_var_trainable(True)
    poseNet.rnn_conv_input()
    loss_tensor, pred_tensor = poseNet.rnn_with_batch_one(label_onehot)
    lgdts_tensor = build_training_ops(loss_tensor)
    
    # Session Saver summary_writer
    sess = tf.Session()
    cpm_saver = tf.train.Saver(var_list=cpm_var)
    cpm_ckpt = tf.train.get_checkpoint_state("logs/")
    if cpm_ckpt:
        cpm_saver.restore(sess, cpm_ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("Tensorflow ckpt not found")
    
    rnn_saver = tf.train.Saver()
    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess, rnn_ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    
    summary_writer = tf.summary.FileWriter("rnn_logs/summary", sess.graph)

    # Load video
    vgen = video_utils.video_frame_class_gen(1, TIME_STEP)
    for itr in range(1, int(1e7)):
        batch_time_frames, batch_time_labels = next(vgen)
        batch_img = np.reshape(batch_time_frames, [TIME_STEP, pa.PH, pa.PW, 3])
        feed_dict = {img_holder: batch_img, label_holder: batch_time_labels}
        loss_num, g_step_num, lr_num, train_op = sess.run(lgdts_tensor[0:4], feed_dict=feed_dict)
        print_log(loss_num, g_step_num, lr_num, itr)
    
        # Summary
        if itr % 100 == 0:
            summary_str = sess.run(lgdts_tensor[4], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, g_step_num)
        
            rnn_saver.save(sess, "rnn_logs/")
            print('Model Saved.')

    sess.close()
    
    pass

if __name__ == "__main__":
    tf.app.run()
exit(0)
