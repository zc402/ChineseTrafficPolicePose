import tensorflow as tf
from tensorflow.contrib import rnn
import parameters as pa


def build_rnn_network(batch_time_input, n_classes, previous_states=None):
    """
    build rnn network
    :param previous_states: States of previous rnn network
    :param n_classes: Number of output classes
    :param batch_time_input: [batch, time_step, n_input]
    :return:prediction_list, state of current rnn
    """

    num_units = 32
    # list [time_step][batch, n_classes]
    input_list = tf.unstack(batch_time_input, axis=1)
    lstm_layer = rnn.BasicLSTMCell(num_units=num_units)
    print(lstm_layer.state_size)
    # outputs: list [time_step][batch, n_units]
    lstm_outputs, last_states = rnn.static_rnn(
        lstm_layer, input_list, initial_state=previous_states, dtype=tf.float32)

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
        out_bias for output in lstm_outputs]

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
    loss = tf.reduce_mean(time_batch_loss_list)
    return loss


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

    head_v = joint_vector(pa.bones[0])
    head_v = tf.expand_dims(head_v, axis=1)
    head_scalar = tf.norm(head_v, axis=2) + 1e-7
    body_v_list = [joint_vector(b) for b in pa.bones[1:]]
    body_v = tf.stack(body_v_list, axis=1)  # I J XY
    body_scalar = tf.norm(body_v, axis=2) + 1e-7  # I J
    # [I][J]
    arm_wrt_head = body_scalar / head_scalar
    # Sine and cosine
    # down_vector: [0, 1] with norm of 1
    a_norm_b_norm = body_scalar  # * tf.norm(down_vector)->1  # I J
    cross_product = body_v[:, :, 0]
    sine = cross_product / a_norm_b_norm  # I J
    dot_product = body_v[:, :, 1]
    cosine = dot_product / a_norm_b_norm  # I J
    # [I][Joint][Features]
    joint_features = tf.stack([arm_wrt_head, sine, cosine], axis=2)  # I J F
    features = tf.reshape(joint_features, [joint_features.get_shape().as_list()[0], -1])  # I F
    return features

