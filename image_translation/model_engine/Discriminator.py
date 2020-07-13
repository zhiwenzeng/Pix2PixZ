import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from .ConvLayer import downsample, upsample


# 鉴定器模型
class Discriminator(Model):

    def __init__(self, input_channel):
        super(Discriminator, self).__init__()
        self.model = self.get_model(input_channel)

    # 给定一个x, 得到一个预测y
    def call(self, x):
        return self.model(x)

    # 感受野为70*70
    def get_model(self, input_channel=1):
        target_channel = 3
        if input_channel == 1:
            target_channel = 2
        # 正态分布初始化
        initializer = tf.random_normal_initializer(0., 0.02)
        # 灰度图
        inp = Input(shape=[None, None, input_channel], name='input_image')
        # 彩色图：一个是预测结果，一个是真实结果
        tar = Input(shape=[None, None, target_channel], name='target_image')

        # 将x进行拼接
        x = layers.concatenate([inp, tar])
        # 卷积+缩小2
        x = downsample(64, 4, False)(x)  # (128, 128, 64)
        x = downsample(128, 4)(x)  # (64, 64, 128)
        x = downsample(256, 4)(x)  # (32, 32, 256)
        # (31, 31, 512)
        x = tf.keras.layers.Conv2D(512, 2, strides=1, kernel_initializer=initializer, use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x, training=True)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # 卷积, 得到30x30的色块
        y = layers.Conv2D(1, 2, strides=1, kernel_initializer=initializer, activation="sigmoid")(x)  # (bs, 30, 30, 1)
        # 返回模型
        return tf.keras.Model([inp, tar], y)