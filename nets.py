import tensorflow as tf
import tensorflow.contrib.layers as layers
import vgg19_trainable as vgg19


class PoseNet:
    def __init__(self):
        self.VGG_PARAM_PATH = "vgg19.npy"
        self.layer_dict = {}
        self.next_layer_input = None  # Input for next self.conv()

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

    def conv(self, filters, kernel_size, name, relu=True):
        if relu:
            c = layers.conv2d(self.next_layer_input, filters, kernel_size, activation_fn=tf.nn.relu, scope=name)
        else:
            c = layers.conv2d(self.next_layer_input, filters, kernel_size, activation_fn=None, scope=name)
        self.layer_dict[name] = c
        self.next_layer_input = c
        return self

    def vgg_10(self, image_input, trainable=False):
        self.layer_dict['image'] = image_input
        vgg_10_layers = vgg19.Vgg10(self.VGG_PARAM_PATH, trainable)
        vgg_10_out = vgg_10_layers.build(image_input)
        return vgg_10_out

    def inference_pose(self, conv_4_2):
        self.layer_dict['conv_4_2'] = conv_4_2
        (self.feed('conv_4_2')
             .conv(256, 3, 'conv_4_3_cpm')
             .conv(128, 3, 'conv_4_4_cpm')

             .conv(128, 3, 'conv_5_1_cpm_l1')
             .conv(128, 3, 'conv_5_2_cpm_l1')
             .conv(128, 3, 'conv_5_3_cpm_l1')
             .conv(512, 1, 'conv_5_4_cpm_l1')
             .conv(5*2, 1, 'conv_5_5_cpm_l1', relu=False))

        (self.feed('conv_4_4_cpm')
             .conv(128, 3, 'conv_5_1_cpm_l2')
             .conv(128, 3, 'conv_5_2_cpm_l2')
             .conv(128, 3, 'conv_5_3_cpm_l2')
             .conv(512, 1, 'conv_5_4_cpm_l2')
             .conv(6, 1, 'conv_5_5_cpm_l2', relu=False))

        (self.concat(['conv_5_5_cpm_l1', 'conv_5_5_cpm_l2', 'conv_4_4_cpm'], 'concat_stage2')
             .conv(128, 7, 'mconv1_stage2_l1')
             .conv(128, 7, 'mconv2_stage2_l1')
             .conv(128, 7, 'mconv3_stage2_l1')
             .conv(128, 7, 'mconv4_stage2_l1')
             .conv(128, 1, 'mconv5_stage2_l1')
             .conv(5*2, 1, 'mconv6_stage2_l1', relu=False))

        (self.feed('concat_stage2')
             .conv(128, 7, 'mconv1_stage2_l2')
             .conv(128, 7, 'mconv2_stage2_l2')
             .conv(128, 7, 'mconv3_stage2_l2')
             .conv(128, 7, 'mconv4_stage2_l2')
             .conv(128, 1, 'mconv5_stage2_l2')
             .conv(6, 1, 'mconv6_stage2_l2', relu=False)
         )

        (self.concat(['mconv6_stage2_l1', 'mconv6_stage2_l2', 'conv_4_4_cpm'], 'concat_stage3')
             .conv(128, 7, 'mconv1_stage3_l1')
             .conv(128, 7, 'mconv2_stage3_l1')
             .conv(128, 7, 'mconv3_stage3_l1')
             .conv(128, 7, 'mconv4_stage3_l1')
             .conv(128, 1, 'mconv5_stage3_l1')
             .conv(5 * 2, 1, 'mconv6_stage3_l1', relu=False))
        (self.feed('concat_stage3')
             .conv(128, 7, 'mconv1_stage3_l2')
             .conv(128, 7, 'mconv2_stage3_l2')
             .conv(128, 7, 'mconv3_stage3_l2')
             .conv(128, 7, 'mconv4_stage3_l2')
             .conv(128, 1, 'mconv5_stage3_l2')
             .conv(6, 1, 'mconv6_stage3_l2', relu=False)
         )

        (self.concat(['mconv6_stage3_l1', 'mconv6_stage3_l2', 'conv_4_4_cpm'], 'concat_stage4')
             .conv(128, 7, 'mconv1_stage4_l1')
             .conv(128, 7, 'mconv2_stage4_l1')
             .conv(128, 7, 'mconv3_stage4_l1')
             .conv(128, 7, 'mconv4_stage4_l1')
             .conv(128, 1, 'mconv5_stage4_l1')
             .conv(5 * 2, 1, 'mconv6_stage4_l1', relu=False))
        (self.feed('concat_stage4')
             .conv(128, 7, 'mconv1_stage4_l2')
             .conv(128, 7, 'mconv2_stage4_l2')
             .conv(128, 7, 'mconv3_stage4_l2')
             .conv(128, 7, 'mconv4_stage4_l2')
             .conv(128, 1, 'mconv5_stage4_l2')
             .conv(6, 1, 'mconv6_stage4_l2', relu=False)
         )

        (self.concat(['mconv6_stage4_l1', 'mconv6_stage4_l2', 'conv_4_4_cpm'], 'concat_stage5')
             .conv(128, 7, 'mconv1_stage5_l1')
             .conv(128, 7, 'mconv2_stage5_l1')
             .conv(128, 7, 'mconv3_stage5_l1')
             .conv(128, 7, 'mconv4_stage5_l1')
             .conv(128, 1, 'mconv5_stage5_l1')
             .conv(5 * 2, 1, 'mconv6_stage5_l1', relu=False))
        (self.feed('concat_stage5')
             .conv(128, 7, 'mconv1_stage5_l2')
             .conv(128, 7, 'mconv2_stage5_l2')
             .conv(128, 7, 'mconv3_stage5_l2')
             .conv(128, 7, 'mconv4_stage5_l2')
             .conv(128, 1, 'mconv5_stage5_l2')
             .conv(6, 1, 'mconv6_stage5_l2', relu=False)
         )

    def loss_l1_l2(self, batch_pcm, batch_paf, batch_size):
        """
        Build loss of network
        :param batch_pcm: [None, H, W, 6]
        :param batch_paf: [None, H, W, 10]
        :return:
        """
        self.layer_dict['pcm_gt'] = batch_pcm
        self.layer_dict['paf_gt'] = batch_paf
        l1s = [self.layer_dict['conv_5_5_cpm_l1']]
        l2s = [self.layer_dict['conv_5_5_cpm_l2']]
        l1s_loss, l2s_loss = [], []
        for layer_name in self.layer_dict.keys():
            if 'mconv6' in layer_name and '_l1' in layer_name:
                l1s.append(self.layer_dict[layer_name])
            if 'mconv6' in layer_name and '_l2' in layer_name:
                l2s.append(self.layer_dict[layer_name])
        for i, l1 in enumerate(l1s):
            loss = tf.nn.l2_loss(l1 - batch_paf) / batch_size
            tf.summary.scalar(name='l1_stage'+str(i+1), tensor=loss)
            l1s_loss.append(loss)
        for i, l2 in enumerate(l2s):
            loss = tf.nn.l2_loss(l2 - batch_pcm) / batch_size
            tf.summary.scalar(name='l2_stage'+str(i+1), tensor=loss)
            l2s_loss.append(loss)
        total_l1_loss = tf.reduce_mean(l1s_loss)
        total_l2_loss = tf.reduce_mean(l2s_loss)
        total_loss = tf.reduce_mean([total_l1_loss, total_l2_loss])
        tf.summary.scalar(name='total_loss', tensor=total_loss)
        return total_loss

    def add_image_summary(self):
        tf.summary.image("S2L1-0", tf.expand_dims(self.layer_dict['mconv6_stage5_l1'][:, :, :, 0], axis=-1))
        tf.summary.image("S2L2-0", tf.expand_dims(self.layer_dict['mconv6_stage5_l2'][:, :, :, 0], axis=-1))
        tf.summary.image("IMAGE", self.layer_dict['image'])
        tf.summary.image("L1-GT", tf.expand_dims(self.layer_dict['paf_gt'][:, :, :, 0], axis=-1))
        tf.summary.image("L2-GT", tf.expand_dims(self.layer_dict['pcm_gt'][:, :, :, 0], axis=-1))


def inference_person(image):
    """Person inference net, return 4 stages for loss computing"""
    with tf.variable_scope('PersonNet'):
        conv1_1 = layers.conv2d(image, 64, 3, activation_fn=tf.nn.relu, scope='conv1_1')
        conv1_2 = layers.conv2d(conv1_1, 64, 3, activation_fn=tf.nn.relu, scope='conv1_2')
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)

        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, activation_fn=tf.nn.relu, scope='conv2_1')
        conv2_2 = layers.conv2d(conv2_1, 128, 3, activation_fn=tf.nn.relu, scope='conv2_2')
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)

        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, activation_fn=tf.nn.relu, scope='conv3_1')
        conv3_2 = layers.conv2d(conv3_1, 256, 3, activation_fn=tf.nn.relu, scope='conv3_2')
        conv3_3 = layers.conv2d(conv3_2, 256, 3, activation_fn=tf.nn.relu, scope='conv3_3')
        conv3_4 = layers.conv2d(conv3_3, 256, 3, activation_fn=tf.nn.relu, scope='conv3_4')
        pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)

        conv4_1 = layers.conv2d(pool3_stage1, 512, 3, activation_fn=tf.nn.relu, scope='conv4_1')
        conv4_2 = layers.conv2d(conv4_1, 512, 3, activation_fn=tf.nn.relu, scope='conv4_2')
        conv4_3 = layers.conv2d(conv4_2, 512, 3, activation_fn=tf.nn.relu, scope='conv4_3')
        conv4_4 = layers.conv2d(conv4_3, 512, 3, activation_fn=tf.nn.relu, scope='conv4_4')
        conv5_1 = layers.conv2d(conv4_4, 512, 3, activation_fn=tf.nn.relu, scope='conv5_1')
        conv5_2_cpm = layers.conv2d(conv5_1, 128, 3, activation_fn=tf.nn.relu, scope='conv5_2_cpm')
        conv6_1_cpm = layers.conv2d(conv5_2_cpm, 512, 1, activation_fn=tf.nn.relu, scope='conv6_1_cpm')
        conv6_2_cpm = layers.conv2d(conv6_1_cpm, 1, 1, activation_fn=None, scope='conv6_2_cpm')

        concat_stage2 = tf.concat(axis=3, values=[conv6_2_cpm, conv5_2_cpm])
        m_conv1_stage2 = layers.conv2d(concat_stage2, 128, 7, activation_fn=tf.nn.relu, scope='m_conv1_stage2')
        m_conv2_stage2 = layers.conv2d(m_conv1_stage2, 128, 7, activation_fn=tf.nn.relu, scope='m_conv2_stage2')
        m_conv3_stage2 = layers.conv2d(m_conv2_stage2, 128, 7, activation_fn=tf.nn.relu, scope='m_conv3_stage2')
        m_conv4_stage2 = layers.conv2d(m_conv3_stage2, 128, 7, activation_fn=tf.nn.relu, scope='m_conv4_stage2')
        m_conv5_stage2 = layers.conv2d(m_conv4_stage2, 128, 7, activation_fn=tf.nn.relu, scope='m_conv5_stage2')
        m_conv6_stage2 = layers.conv2d(m_conv5_stage2, 128, 1, activation_fn=tf.nn.relu, scope='m_conv6_stage2')
        m_conv7_stage2 = layers.conv2d(m_conv6_stage2, 1, 1, activation_fn=None, scope='m_conv7_stage2')

        concat_stage3 = tf.concat(axis=3, values=[m_conv7_stage2, conv5_2_cpm])
        m_conv1_stage3 = layers.conv2d(concat_stage3,  128, 7, activation_fn=tf.nn.relu, scope='m_conv1_stage3')
        m_conv2_stage3 = layers.conv2d(m_conv1_stage3, 128, 7, activation_fn=tf.nn.relu, scope='m_conv2_stage3')
        m_conv3_stage3 = layers.conv2d(m_conv2_stage3, 128, 7, activation_fn=tf.nn.relu, scope='m_conv3_stage3')
        m_conv4_stage3 = layers.conv2d(m_conv3_stage3, 128, 7, activation_fn=tf.nn.relu, scope='m_conv4_stage3')
        m_conv5_stage3 = layers.conv2d(m_conv4_stage3, 128, 7, activation_fn=tf.nn.relu, scope='m_conv5_stage3')
        m_conv6_stage3 = layers.conv2d(m_conv5_stage3, 128, 1, activation_fn=tf.nn.relu, scope='m_conv6_stage3')
        m_conv7_stage3 = layers.conv2d(m_conv6_stage3, 1, 1, activation_fn=None, scope='m_conv7_stage3')

        concat_stage4 = tf.concat(axis=3, values=[m_conv7_stage3, conv5_2_cpm])
        m_conv1_stage4 = layers.conv2d(concat_stage4,  128, 7, activation_fn=tf.nn.relu, scope='m_conv1_stage4')
        m_conv2_stage4 = layers.conv2d(m_conv1_stage4, 128, 7, activation_fn=tf.nn.relu, scope='m_conv2_stage4')
        m_conv3_stage4 = layers.conv2d(m_conv2_stage4, 128, 7, activation_fn=tf.nn.relu, scope='m_conv3_stage4')
        m_conv4_stage4 = layers.conv2d(m_conv3_stage4, 128, 7, activation_fn=tf.nn.relu, scope='m_conv4_stage4')
        m_conv5_stage4 = layers.conv2d(m_conv4_stage4, 128, 7, activation_fn=tf.nn.relu, scope='m_conv5_stage4')
        m_conv6_stage4 = layers.conv2d(m_conv5_stage4, 128, 1, activation_fn=tf.nn.relu, scope='m_conv6_stage4')
        m_conv7_stage4 = layers.conv2d(m_conv6_stage4, 1, 1, activation_fn=None, scope='m_conv7_stage4')

        return [conv6_2_cpm, m_conv7_stage2, m_conv7_stage3, m_conv7_stage4]


class PersonPredictor:
    def __init__(self, input_image):
        self.input_image = input_image
        self.stage_losses = None
        self.total_loss = None  # Loss to be optimized
        self.heatmaps = inference_person(input_image)
        self.output = self.heatmaps[-1]
        self.output_shape = self.output.get_shape().as_list()
        self.loss_summary = None
        self.img_summary = None

    def build_loss(self, heatmap_gt, batch_size):
        self.stage_losses = []
        for stage, heatmap in enumerate(self.heatmaps):
            stage_loss = tf.nn.l2_loss(heatmap - heatmap_gt) / batch_size
            self.stage_losses.append(stage_loss)
            tf.summary.scalar("stage"+str(stage), stage_loss)
        self.total_loss = tf.reduce_mean(self.stage_losses)

    def add_img_summary(self):
        tf.summary.image("image_in", self.input_image, 3, collections=["img"])
        tf.summary.image("heatmap_out", self.output, 3, collections=["img"])
        img_summary_op = tf.summary.merge_all(key="img")
        return img_summary_op


def add_gradient_summary(grads):
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)

