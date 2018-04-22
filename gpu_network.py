import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn

class PoseNet:
    def __init__(self):
        print("Don't Forget To Set Input Scale To .0 ~ .1")
        self.layer_dict = {}
        self.next_layer_input = None  # Input for next self.conv()
        self.var_trainable = True
    
    def set_var_trainable(self, trainable):
        """
        Toggle variants between trainable and un-trainable
        :param trainable: bool
        """
        self.var_trainable = trainable

    def feed(self, feed_name):
        try:
            feed_layer = self.layer_dict[feed_name]
        except KeyError:
            raise KeyError('Unknown layer name fed: %s' % feed_name)
        self.next_layer_input = feed_layer
        return self

    def concat(self, concat_list, concat_layer_name):
        val = []
        for concat_name in concat_list:
            try:
                val.append(self.layer_dict[concat_name])
            except KeyError:
                raise KeyError('Unknown layer name fed: %s' % concat_name)
        c = tf.concat(values=val, axis=3, name=concat_layer_name)
        self.layer_dict[concat_layer_name] = c
        self.next_layer_input = c
        return self

    def conv(self, filters, kernel_size, name, relu=True, padding='SAME'):
        if relu:
            c = layers.conv2d(self.next_layer_input, filters, kernel_size, activation_fn=tf.nn.relu, scope=name, trainable=self.var_trainable, padding=padding)
        else:
            c = layers.conv2d(self.next_layer_input, filters, kernel_size, activation_fn=None, scope=name, trainable=self.var_trainable, padding='SAME')
        self.layer_dict[name] = c
        self.next_layer_input = c
        return self

    def max_pool(self, name):
        p = layers.max_pool2d(self.next_layer_input, 2, 2)
        self.layer_dict[name] = p
        self.next_layer_input = p
        return self

    def inference_paf_pcm(self, image_input):
        """
        Inference l1:paf l2:pcm heatmaps
        :param image_input: Original image
        :return: Concatenation of [paf pcm]
        """
        self.layer_dict['image'] = image_input
        (self.feed('image')
             .conv(64, 3, 'conv_1_1')
             .conv(64, 3, 'conv_1_2')
             .max_pool('pool_1_stage1')
             .conv(128, 3, 'conv_2_1')
             .conv(128, 3, 'conv_2_2')
             .max_pool('pool_2_stage1')
             .conv(256, 3, 'conv_3_1')
             .conv(256, 3, 'conv_3_2')
             .conv(256, 3, 'conv_3_3')
             .conv(256, 3, 'conv_3_4')
             .max_pool('pool_3_stage1')
             .conv(512, 3, 'conv_4_1')
             .conv(512, 3, 'conv_4_2')
            # Top 10 layers ended
             .conv(256, 3, 'conv_4_3_cpm')
             .conv(128, 3, 'conv_4_4_cpm')

             .conv(128, 3, 'conv_5_1_cpm_l1')
             .conv(128, 3, 'conv_5_2_cpm_l1')
             .conv(128, 3, 'conv_5_3_cpm_l1')
             .conv(512, 1, 'conv_5_4_cpm_l1')
             .conv(7*2, 1, 'conv_5_5_cpm_l1', relu=False))

        (self.feed('conv_4_4_cpm')
             .conv(128, 3, 'conv_5_1_cpm_l2')
             .conv(128, 3, 'conv_5_2_cpm_l2')
             .conv(128, 3, 'conv_5_3_cpm_l2')
             .conv(512, 1, 'conv_5_4_cpm_l2')
             .conv(8, 1, 'conv_5_5_cpm_l2', relu=False))

        (self.concat(['conv_5_5_cpm_l1', 'conv_5_5_cpm_l2', 'conv_4_4_cpm'], 'concat_stage2')
             .conv(128, 7, 'mconv1_stage2_l1')
             .conv(128, 7, 'mconv2_stage2_l1')
             .conv(128, 7, 'mconv3_stage2_l1')
             .conv(128, 7, 'mconv4_stage2_l1')
             .conv(128, 7, 'mconv5_stage2_l1')
             .conv(128, 1, 'mconv6_stage2_l1')
             .conv(7*2, 1, 'mconv7_stage2_l1', relu=False))

        (self.feed('concat_stage2')
             .conv(128, 7, 'mconv1_stage2_l2')
             .conv(128, 7, 'mconv2_stage2_l2')
             .conv(128, 7, 'mconv3_stage2_l2')
             .conv(128, 7, 'mconv4_stage2_l2')
             .conv(128, 7, 'mconv5_stage2_l2')
             .conv(128, 1, 'mconv6_stage2_l2')
             .conv(8, 1, 'mconv7_stage2_l2', relu=False)
         )

        (self.concat(['mconv7_stage2_l1', 'mconv7_stage2_l2', 'conv_4_4_cpm'], 'concat_stage3')
             .conv(128, 7, 'mconv1_stage3_l1')
             .conv(128, 7, 'mconv2_stage3_l1')
             .conv(128, 7, 'mconv3_stage3_l1')
             .conv(128, 7, 'mconv4_stage3_l1')
             .conv(128, 7, 'mconv5_stage3_l1')
             .conv(128, 1, 'mconv6_stage3_l1')
             .conv(7*2, 1, 'mconv7_stage3_l1', relu=False))
        (self.feed('concat_stage3')
             .conv(128, 7, 'mconv1_stage3_l2')
             .conv(128, 7, 'mconv2_stage3_l2')
             .conv(128, 7, 'mconv3_stage3_l2')
             .conv(128, 7, 'mconv4_stage3_l2')
             .conv(128, 7, 'mconv5_stage3_l2')
             .conv(128, 1, 'mconv6_stage3_l2')
             .conv(8, 1, 'mconv7_stage3_l2', relu=False)
         )

        self.concat(['mconv7_stage3_l1', 'mconv7_stage3_l2'], 'paf_pcm_output')
        return self.layer_dict['paf_pcm_output']

    def _loss_paf_pcm(self, batch_pcm, batch_paf):
        """
        Use pose inference to build loss of network
        :param batch_pcm: [None, H, W, 8]
        :param batch_paf: [None, H, W, 12]
        :return: total loss value
        """
        batch_size = batch_pcm.get_shape().as_list()[0]
        self.layer_dict['pcm_gt'] = batch_pcm
        self.layer_dict['paf_gt'] = batch_paf
        l1s = [self.layer_dict['conv_5_5_cpm_l1']]
        l2s = [self.layer_dict['conv_5_5_cpm_l2']]
        l1s_loss, l2s_loss = [], []
        for layer_name in self.layer_dict.keys():
            if 'mconv7' in layer_name and '_l1' in layer_name:
                l1s.append(self.layer_dict[layer_name])
            if 'mconv7' in layer_name and '_l2' in layer_name:
                l2s.append(self.layer_dict[layer_name])
        for i, l1 in enumerate(l1s):
            loss = tf.nn.l2_loss(l1 - batch_paf) / batch_size  / 14
            tf.summary.scalar(name='l1_stage'+str(i+1), tensor=loss)
            l1s_loss.append(loss)
        for i, l2 in enumerate(l2s):
            loss = tf.nn.l2_loss(l2 - batch_pcm) / batch_size / 8
            tf.summary.scalar(name='l2_stage'+str(i+1), tensor=loss)
            l2s_loss.append(loss)
        total_l1_loss = tf.reduce_mean(l1s_loss)
        total_l2_loss = tf.reduce_mean(l2s_loss)
        total_loss = tf.reduce_mean([total_l1_loss, total_l2_loss])
        tf.summary.scalar(name='total_loss', tensor=total_loss)
        return total_loss

    def _add_paf_summary(self):
        """
        Add images of paf_pcm to tensorboard
        """
        tf.summary.image("L1-Final", tf.expand_dims(self.layer_dict['mconv7_stage3_l1'][:, :, :, 0], axis=-1))
        tf.summary.image("L2-Final", tf.expand_dims(self.layer_dict['mconv7_stage3_l2'][:, :, :, 0], axis=-1))
        tf.summary.image("IMAGE", self.layer_dict['image'])
        tf.summary.image("L1-GT", tf.expand_dims(self.layer_dict['paf_gt'][:, :, :, 0], axis=-1))
        tf.summary.image("L2-GT", tf.expand_dims(self.layer_dict['pcm_gt'][:, :, :, 0], axis=-1))
    
    def build_paf_pcm_loss(self, img_tensor, i_hv_tensor):
        """
        Build the loss of paf_pcm, only used on training paf_pcm layers
        :param img_tensor:
        :param i_hv_tensor:
        :return: A scalar loss tensor
        """
        self.inference_paf_pcm(img_tensor)
        total_loss = self._loss_paf_pcm(i_hv_tensor[:, :, :, :8], i_hv_tensor[:, :, :, 8:])
        self._add_paf_summary()
        return total_loss
        
    def rnn_conv_input(self): # TODO: don't forget to put trainable before this layer to false
        """
        Conv layers as the input of rnn network
        :return: [B, 1, 1, 40] tensor with 40 features for each image
        """
        raise RuntimeError('Deprecated! will be removed')
        paf_pcm = self.layer_dict['paf_pcm_output']
        pcm = paf_pcm[:, :, :, 10:16]
        self.layer_dict['pcm_output'] = pcm # Only use part confidence map
        assert(paf_pcm is not None, 'Build CPM network before calling rnn conv!')
        assert(paf_pcm.get_shape().as_list()[1:4] == [64, 64, 16])
        (self.feed('pcm_output')
         .max_pool('pool_1_rnn') # 32
         .conv(16, 3, 'rconv2')
         .max_pool('pool_2_rnn') # 16
         .conv(16, 3, 'rconv3')
         .max_pool('pool_3_rnn') # 8
         .conv(16, 3, 'rconv5')
         .max_pool('pool_4_rnn') # 4
         .conv(16, 4, 'rconv7', padding='VALID')) # [B, 1, 1, 40]
        assert(self.layer_dict['rconv7'].get_shape().as_list()[1:3] == [1, 1])
        return self.layer_dict['rconv7']
    
    def rnn_with_batch_one(self, batch_time_class):
        rconv7 = self.layer_dict['rconv7']
        img_batch_size = rconv7.get_shape().as_list()[0]
        img_features = tf.reshape(rconv7, [1, img_batch_size, -1]) # time, consecutive images, features
        return build_rnn_network(img_features, batch_time_class)
        
        
        
def build_rnn_network(batch_time_input, batch_time_class):
    """
    build rnn network
    :param batch_time_input: [batch, time_step, n_input]
    :param batch_time_class: [batch, time_step, n_classes]
    :return:loss, prediction_list
    """
    n_classes = batch_time_class.get_shape().as_list()[2]
    num_units = 32
    assert(batch_time_input.get_shape().as_list()[1] == batch_time_class.get_shape().as_list()[1])
    input_list = tf.unstack(batch_time_input, axis=1) # list [time_step][batch, n_classes]
    lstm_layer = rnn.BasicLSTMCell(num_units=num_units)
    lstm_outputs, _ = rnn.static_rnn(lstm_layer, input_list, dtype=tf.float32) # list [time_step][batch, n_units]
    
    # Fully connect outputs to class_num
    # weights and biases of fully connected layer
    out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
    out_bias=tf.Variable(tf.random_normal([n_classes]))
    
    # Each output multiply by same fc layer:  list [time_step][batch, n_outputs]
    lstm_prediction_list = [tf.matmul(output,out_weights) + out_bias for output in lstm_outputs]
    # Labels list: [t][b, classes]
    t_bc_label_list = tf.unstack(batch_time_class, axis=1)
    time_batch_loss_list = []
    for i in range(len(lstm_prediction_list)):
        time_batch_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=lstm_prediction_list[i], labels=t_bc_label_list[i])
        time_batch_loss_list.append(time_batch_loss)
    loss = tf.reduce_mean(time_batch_loss_list)
    return loss, lstm_prediction_list
