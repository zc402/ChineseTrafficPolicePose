import parameters as pa
import tensorflow as tf
import gpu_network
import rnn_network
import cv2
import numpy as np

def _parse_joint(pcm):
    "Return joints[J, XY]"
    joint_parts = []  # 8 joint parts
    for idx_joint in range(8):
        heat = pcm[0, :, :, idx_joint]
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
        joint_parts.append(j_xy)  # [j][XY]
    
    # video_utils.save_joints_position(v_name)
    joint_xy = np.asarray(joint_parts)
    return joint_xy

def build_evaluation_network():
    g_1 = tf.Graph()
    g_2 = tf.Graph()
    with g_1.as_default():
        batch_size = 1
        
        # Place Holder
        img_holder = tf.placeholder(tf.float32, [batch_size, pa.PH, pa.PW, 3])
        # Entire network
        paf_pcm_tensor = gpu_network.PoseNet().inference_paf_pcm(img_holder)
        saver = tf.train.Saver()
    
    with g_2.as_default():
        NUM_CLASSES = 9  # 8 classes + 1 for no move
        NUM_JOINTS = 8
        
        # batch_time_joint_holder:
        btjh = tf.placeholder(tf.float32, [1, 1, NUM_JOINTS, 2])  # 2:xy
        c_state_holder = tf.placeholder(tf.float32, [32])
        h_state_holder = tf.placeholder(tf.float32, [32])
        
        state_tuple = ([c_state_holder], [h_state_holder])
        
        with tf.variable_scope("rnn-net"):
            # b t c(0/1)
            img_j_xy = tf.reshape(btjh, [-1, NUM_JOINTS, 2])
            img_fe = rnn_network.extract_features_from_joints(img_j_xy)
            btf = tf.reshape(img_fe, [1, 1, -1])
            pred, state = rnn_network.build_rnn_network(btf, NUM_CLASSES, state_tuple)
            # model evaluation
            btc_pred = tf.transpose(pred, [1, 0, 2])  # TBC to BTC
            btc_pred_max = tf.argmax(btc_pred, 2)
            rnn_saver = tf.train.Saver()

    rnn_saved_state = [np.zeros(32), np.zeros(32)]
    # Restore CPM and LSTM
    sess1 = tf.Session(graph=g_1)
    sess2 = tf.Session(graph=g_2)

    ckpt = tf.train.get_checkpoint_state("logs/")
    if ckpt:
        saver.restore(sess1, ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("CPM ckpt not found.")

    rnn_ckpt = tf.train.get_checkpoint_state("rnn_logs/")
    if rnn_ckpt:
        rnn_saver.restore(sess2, rnn_ckpt.model_checkpoint_path)
    else:
        raise FileNotFoundError("RNN ckpt not found.")
    
    # Run evaluation for a frame
    def evaluate(frame):
        
        if frame is None:
            sess1.close()
            sess2.close()
            return None
            
        # Closure function. sessions and tensors is preserved globally
        feed_dict = {img_holder: frame}
        paf_pcm = sess1.run(paf_pcm_tensor, feed_dict=feed_dict)
        pcm = paf_pcm[:, :, :, 14:]
        np.clip(pcm, 0., 1., pcm)
        joint_xy = _parse_joint(pcm)
        joint_data = joint_xy[np.newaxis, np.newaxis, :, :]
        
        nonlocal rnn_saved_state
        feed_dict = {btjh: joint_data, c_state_holder: rnn_saved_state[0], h_state_holder: rnn_saved_state[1]}
        # Return: prediction, rnn previous state, 18 features
        btc_pred_num, state_num, lsc18 = sess2.run([btc_pred_max, state, img_fe], feed_dict=feed_dict)
        
        rnn_saved_state = [state_num[0][0], state_num[1][0]]
        pred = np.reshape(btc_pred_num, [-1])
        
        return pred, pcm, joint_xy, lsc18
            
    return evaluate


def result_analyzer():
    """
    Initialize result analyzer
    :return: method to draw analyzed picture
    """
    final_out = np.zeros([512 * 2, 512 * 2, 3], dtype=np.uint8)
    
    skeleton_out = final_out[0:512, 512:1024, :]
    ana_out = final_out[512:1024, 0:512, :]
    pose_name_out = final_out[512:1024, 512:1024, :]

    bone_colors = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (64, 255, 0),
                   (0, 255, 255), (0, 0, 255), (255, 0, 255)]
    bone_colors_bgr = [(b, g, r) for r, g, b in bone_colors]
    
    def analytic_picture(frame, pred, pcm, joint_xy, lsc18):
        """
        Return picture with analytic information
        :param frame: video frame being analyzed. shape:[512, 512, 3]
        :param pred: predicted result
        :param pcm: part confidence map
        :param joint_xy: joint coordinates
        :param lsc18: 18 features
        :return: picture of size [512*2, 512*2, 3]
        """
        final_out.fill(0)
        
        # Heatmap image
        heatmap_out = np.sum(pcm[0], axis=2) * 255.
        heatmap_out.astype(np.uint8)
        # Skeleton image
        
        # Connection of bones. Inside one image
        for b_num, (b1, b2) in enumerate(pa.bones):
            if np.less(
                joint_xy[b1, :],
                0).any() or np.less(
                joint_xy[b2, :],
                0).any():
                continue  # no detection
            x1 = int(joint_xy[b1, 0] * 512)
            y1 = int(joint_xy[b1, 1] * 512)
            x2 = int(joint_xy[b2, 0] * 512)
            y2 = int(joint_xy[b2, 1] * 512)
            cv2.line(skeleton_out, (x1, y1), (x2, y2), bone_colors_bgr[b_num], 4)
        
        # Draw Extracted 18 features
        cv2.putText(ana_out, "Relative Length of Bones", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for n, l in enumerate(np.concatenate(([1.0], lsc18[0, ::3]))):
            x2 = x1 = 20 + n * 30
            y1 = 60
            y2 = int(l * 100.) + y1
            cv2.line(ana_out, (x1, y1), (x2, y2), bone_colors_bgr[n], 4)
            cv2.putText(ana_out, "%1.1f" % l, (x2 - 10, y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1)
        
        cv2.putText(ana_out, "Sine Values", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for n, l in enumerate(lsc18[0, 1::3]):
            x2 = x1 = 20 + n * 30
            y1 = 270 + 50
            y2 = int(l * 50.) + y1
            cv2.line(ana_out, (x1, y1), (x2, y2), bone_colors_bgr[n + 1], 4)
            cv2.putText(ana_out, "%1.1f" % l, (x2 - 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1)
        
        cv2.putText(ana_out, "Cosine Values", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for n, l in enumerate(lsc18[0, 2::3]):
            x2 = x1 = 250 + n * 30
            y1 = 270 + 50
            y2 = int(l * 50.) + y1
            cv2.line(ana_out, (x1, y1), (x2, y2), bone_colors_bgr[n + 1], 4)
            cv2.putText(ana_out, "%1.1f" % l, (x2 - 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255),
                        1)
        pred_text = pa.police_dict[pred[0]]
        print(pred_text)
        cv2.putText(pose_name_out, pred_text, (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        final_out[0:512, 0:512, :] = frame
        return final_out
    return analytic_picture