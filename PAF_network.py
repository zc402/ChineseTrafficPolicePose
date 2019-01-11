import tensorflow as tf
from tensorflow.contrib import layers


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
             .conv(11*2, 1, 'conv_5_5_cpm_l1', relu=False))

        (self.feed('conv_4_4_cpm')
             .conv(128, 3, 'conv_5_1_cpm_l2')
             .conv(128, 3, 'conv_5_2_cpm_l2')
             .conv(128, 3, 'conv_5_3_cpm_l2')
             .conv(512, 1, 'conv_5_4_cpm_l2')
             .conv(14, 1, 'conv_5_5_cpm_l2', relu=False))

        (self.concat(['conv_5_5_cpm_l1', 'conv_5_5_cpm_l2', 'conv_4_4_cpm'], 'concat_stage2')
             .conv(128, 7, 'mconv1_stage2_l1')
             .conv(128, 7, 'mconv2_stage2_l1')
             .conv(128, 7, 'mconv3_stage2_l1')
             .conv(128, 7, 'mconv4_stage2_l1')
             .conv(128, 7, 'mconv5_stage2_l1')
             .conv(128, 1, 'mconv6_stage2_l1')
             .conv(11*2, 1, 'mconv7_stage2_l1', relu=False))

        (self.feed('concat_stage2')
             .conv(128, 7, 'mconv1_stage2_l2')
             .conv(128, 7, 'mconv2_stage2_l2')
             .conv(128, 7, 'mconv3_stage2_l2')
             .conv(128, 7, 'mconv4_stage2_l2')
             .conv(128, 7, 'mconv5_stage2_l2')
             .conv(128, 1, 'mconv6_stage2_l2')
             .conv(14, 1, 'mconv7_stage2_l2', relu=False)
         )

        (self.concat(['mconv7_stage2_l1', 'mconv7_stage2_l2', 'conv_4_4_cpm'], 'concat_stage3')
             .conv(128, 7, 'mconv1_stage3_l1')
             .conv(128, 7, 'mconv2_stage3_l1')
             .conv(128, 7, 'mconv3_stage3_l1')
             .conv(128, 7, 'mconv4_stage3_l1')
             .conv(128, 7, 'mconv5_stage3_l1')
             .conv(128, 1, 'mconv6_stage3_l1')
             .conv(11*2, 1, 'mconv7_stage3_l1', relu=False))
        (self.feed('concat_stage3')
             .conv(128, 7, 'mconv1_stage3_l2')
             .conv(128, 7, 'mconv2_stage3_l2')
             .conv(128, 7, 'mconv3_stage3_l2')
             .conv(128, 7, 'mconv4_stage3_l2')
             .conv(128, 7, 'mconv5_stage3_l2')
             .conv(128, 1, 'mconv6_stage3_l2')
             .conv(14, 1, 'mconv7_stage3_l2', relu=False)
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
            loss = tf.nn.l2_loss(l1 - batch_paf) / batch_size / (11*2)
            tf.summary.scalar(name='l1_stage'+str(i+1), tensor=loss)
            l1s_loss.append(loss)
        for i, l2 in enumerate(l2s):
            loss = tf.nn.l2_loss(l2 - batch_pcm) / batch_size / 14
            tf.summary.scalar(name='l2_stage'+str(i+1), tensor=loss)
            l2s_loss.append(loss)
        total_l1_loss = tf.reduce_mean(l1s_loss)
        total_l2_loss = tf.reduce_mean(l2s_loss)
        total_loss = tf.reduce_mean([total_l1_loss, total_l2_loss])
        tf.summary.scalar(name='total_loss', tensor=total_loss)
        return total_loss

    def _add_pcm_paf_summary(self):
        """
        Add images of paf_pcm to tensorboard
        """
        tf.summary.image("L1-Final", tf.reduce_max(self.layer_dict['mconv7_stage3_l1'], axis=3, keepdims=True))
        tf.summary.image("L2-Final", tf.reduce_max(self.layer_dict['mconv7_stage3_l2'], axis=3, keepdims=True))
        tf.summary.image("IMAGE", self.layer_dict['image'])
        tf.summary.image("L1-GT", tf.reduce_max(self.layer_dict['paf_gt'], axis=3, keepdims=True))
        tf.summary.image("L2-GT", tf.reduce_max(self.layer_dict['pcm_gt'], axis=3, keepdims=True))
    
    def build_paf_pcm_loss(self, img_nhwc, pcm_nhwc, paf_nhwc):
        """
        Build the loss of paf_pcm, only used on training paf_pcm layers
        :param img_nhwc:
        :param i_hv_tensor:
        :return: A scalar loss tensor
        """
        self.inference_paf_pcm(img_nhwc)
        total_loss = self._loss_paf_pcm(pcm_nhwc, paf_nhwc)
        self._add_pcm_paf_summary()
        return total_loss
        



