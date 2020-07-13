import tensorflow as tf
from tensorflow.keras import Model, Input, layers
from .ConvLayer import downsample, upsample, downsample_gen


# 生成器模式
class Generator(Model):

    def __init__(self, input_channel):
        super(Generator, self).__init__()
        self.model = self.get_model(input_channel)

    # 自定义模型的回调, 给定一个参数x, 得到结果y, y为预测结果
    def call(self, x):
        return self.model(x)

    # 获得模型的方法, U-NET改进网络
    def get_model(self, input_channel=1):
        output_channel = 3
        if input_channel == 1:
            output_channel = 2
        # 8层的卷积+缩小
        downs = [
            downsample_gen(64, 4, apply_batchnorm=False),  # (64, 128, 128)
            downsample_gen(128, 4),  # (128, 64, 64)
            downsample_gen(256, 4),  # (256, 32, 32)
            downsample_gen(512, 4),  # (512, 16, 16)
            downsample_gen(512, 4),  # (512, 8, 8)
            downsample_gen(512, 4),  # (512, 4, 4)
            downsample_gen(512, 4),  # (512, 2, 2)
            downsample_gen(512, 4),  # (512, 1, 1)
        ]

        # 7层的反卷积+扩大
        ups = [
            upsample(512, 4, apply_dropout=True),  # (512, 2, 2)
            upsample(512, 4, apply_dropout=True),  # (512, 4, 4)
            upsample(512, 4, apply_dropout=True),  # (512, 8, 8)
            upsample(512, 4),  # (512, 16, 16)
            upsample(256, 4),  # (256, 32, 32)
            upsample(128, 4),  # (128, 64, 64)
            upsample(64, 4),  # (64, 128, 128)
        ]

        # 正态分布初始化
        initializer = tf.random_normal_initializer(0., 0.02)
        # 最后一层的反卷积, 得到生成的图片
        last = layers.Conv2DTranspose(output_channel, 4, strides=2, padding='same',
                                      kernel_initializer=initializer, activation='tanh')

        # 由于黑白图片, 所以形状应该是(width, height, input_channel)
        inputs = Input(shape=[None, None, input_channel])

        # 接下来进行x的运算
        x = inputs
        # 存储每一层卷积的结果
        dres = []

        # 每一层的卷积
        for down in downs:
            x = down(x)
            dres.append(x)

        # 由于U-NET的特点, 应该是左右两边依次向中间靠拢拼接
        # 只需要前7层, 因为最后一层是共有的
        dres = reversed(dres[:-1])
        for up, res in zip(ups, dres):
            x = up(x)
            x = layers.Concatenate()([x, res])
        # 最后一层的反卷积, 得到图片(width, height, 3)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        y = last(x)
        # 返回模型
        return Model(inputs, y)
