import tensorflow as tf
from tensorflow.contrib import rnn
import parameters as pa
import numpy as np

def build_rnn_network(batch_time_input, n_classes, training, previous_states=None):
    """
    build rnn network
    :param previous_states: States of previous rnn network
    :param n_classes: Number of output classes
    :param batch_time_input: [batch, time_step, n_input]
    :return:prediction_list, state of current rnn
    """

    num_units = pa.RNN_HIDDEN_UNITS
    # list [time_step][batch, n_classes]
    input_list = tf.unstack(batch_time_input, axis=1)
    lstm_layer = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
    print(lstm_layer.state_size)
    # outputs: list [time_step][batch, n_units]
    lstm_outputs, last_states = rnn.static_rnn(
        lstm_layer, input_list, initial_state=previous_states, dtype=tf.float32)

    # Dropout
    lstm_outputs_dp = [tf.layers.dropout(output, training=training) for output in lstm_outputs]

    # Fully connect outputs to class_num
    # weights and biases of fully connected layer
    out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
    out_bias = tf.Variable(tf.random_normal([n_classes]))

    # Each output multiply by same fc layer:  list [time_step][batch,
    # n_outputs]
    lstm_prediction_list = [
        tf.matmul(
            output,
            out_weights) +
        out_bias for output in lstm_outputs_dp]

    return lstm_prediction_list, last_states


def build_rnn_loss(lstm_prediction_list, batch_time_class_label):
    """
    Build rnn loss tensor
    :param lstm_prediction_list: list [time_step][batch, n_outputs]
    :param batch_time_class_label: [batch, time_step, n_classes]
    :return: total loss
    """
    # n_classes = batch_time_class_label.get_shape().as_list()[2]

    # Labels list: [t][b, classes]
    t_bc_label_list = tf.unstack(batch_time_class_label, axis=1)
    time_batch_loss_list = []
    for i in range(len(lstm_prediction_list)):
        time_batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=lstm_prediction_list[i], labels=t_bc_label_list[i])
        time_batch_loss_list.append(time_batch_loss)
    # Do not compute the loss at the beginning of video
    loss = tf.reduce_mean(time_batch_loss_list[pa.LABEL_DELAY_FRAMES:])
    return loss


def _extract_length_angle_from_sequence(tjc):
    tjc = np.asarray(tjc)
    v_len = tjc.shape[0]
    assert v_len > 0

    time_feature = []
    for time in range(v_len):
        features_list = []
        joint_coor = tjc[time]  # jc for 1 frame, contains all joint positions
        def occluded(b1, b2):
            if np.less(joint_coor[b1, :], 0).any() or np.less(joint_coor[b2, :], 0).any():  # At least 1 part is not visible
                return True
        # Head
        head_b1, head_b2 = pa.bones_head[0]
        if occluded(head_b1, head_b2):
            # Head occluded
            head_norm = 1.
        else:
            head_norm = np.linalg.norm(joint_coor[head_b1, :] - joint_coor[head_b2, :]) + 1e-7

        # Body
        list_bone_length = []
        list_joint_angle = []
        for b_num, (b1, b2) in enumerate(pa.bones_body):
            coor1 = joint_coor[b1, :]
            coor2 = joint_coor[b2, :]
            # At least 1 part is not visible
            if occluded(b1, b2):
                # bone length for (b1, b2) = 0
                # joint angle for (b1, b2) = (sin)0, (cos)0
                list_bone_length.append(0)
                list_joint_angle.append(0)
                list_joint_angle.append(0)
            else:  # Both parts are visible
                bone_vec = coor1 - coor2
                bone_norm = np.linalg.norm(bone_vec) + 1e-7
                bone_cross = np.cross(bone_vec, (0, 1))
                bone_dot = np.dot(bone_vec, (0, 1))
                bone_sin = np.true_divide(bone_cross, bone_norm)
                bone_cos = np.true_divide(bone_dot, bone_norm)
                # wrt_h : With respect to head length
                len_wrt_h = np.true_divide(bone_norm, head_norm)
                list_bone_length.append(len_wrt_h)
                list_joint_angle.append(bone_sin)
                list_joint_angle.append(bone_cos)
        features_list.extend(list_bone_length)
        features_list.extend(list_joint_angle)
        time_feature.append(features_list)
    return np.asarray(time_feature)


def extract_bone_length_joint_angle(btjc):
    """
    Produce batch_time_feature array
    :param btjc:
    :return:
    """
    batch_size = btjc.shape[0]
    batch_time_feature = []
    for i in range(batch_size):
        tjc = btjc[i]
        time_feature = _extract_length_angle_from_sequence(tjc)
        batch_time_feature.append(time_feature)
        print("Generating feature: %d / %d" % (i+1, batch_size))
    return np.asarray(batch_time_feature)


def extract_features_from_joints(joint_tensor):
    """
    Extract features from joint positions.
    Features: limb length wrt head length. sine. cosine.
    :param joint_tensor: [I][Joint][XY]
    :return: [I][F]
    """
    # Joints: 0-r_arm, 1-r_elbow, 2-r_shoulder, 3-l_shoulder, 4-l_elbow,
    # 5-l_arm, 6-head_top, 7-neck
    # bones: 0:head, 1-2-3:right_arm, 4-5-6:left_arm
    def joint_vector(bone):  # return 2 dim [I][XY]
        assert len(bone) == 2
        return joint_tensor[:, bone[1], :] - joint_tensor[:, bone[0], :]
    def xandy_visible(j):
        return tf.logical_and(tf.greater(joint_tensor[:, j, 0], 0.), tf.greater(joint_tensor[:, j, 1], 0.))
    bone_mask = [tf.logical_and(xandy_visible(j1), xandy_visible(j2)) for j1, j2 in pa.bones]  #list[b] [I]
    bone_mask = tf.stack(bone_mask, axis=1) # I B
    bone_mask = tf.expand_dims(bone_mask, axis=-1) # I B 1
    bone_vector = [joint_vector(b) for b in pa.bones] # list[b] [I][XY]
    bone_vector = tf.stack(bone_vector, axis=1) # I B XY
    bone_vector = bone_vector * tf.cast(bone_mask, tf.float32)  # I B XY


    head_v = bone_vector[:, 0, :] # I XY
    head_v = tf.expand_dims(head_v, axis=1) # I 1 XY
    head_scalar = tf.norm(head_v, axis=2) + 1e-7
    body_v = bone_vector[:, 1:, :]  # I B XY
    body_scalar = tf.norm(body_v, axis=2) + 1e-7  # I B
    # [I][J]
    arm_wrt_head = body_scalar / head_scalar
    # Sine and cosine
    # down_vector: [0, 1] with norm of 1
    a_norm_b_norm = body_scalar  # * tf.norm(down_vector)->1  # I J
    cross_product = body_v[:, :, 0]
    sine = cross_product / a_norm_b_norm  # I B
    dot_product = body_v[:, :, 1]
    cosine = dot_product / a_norm_b_norm  # I B
    # [I][Joint][Features]
    joint_features = tf.stack([arm_wrt_head, sine, cosine], axis=2)  # I B F
    features = tf.reshape(joint_features, [joint_features.get_shape().as_list()[0], -1])  # I F
    # features = tf.Print(features, [features], summarize=100)
    return features
