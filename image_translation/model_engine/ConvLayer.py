import tensorflow as tf

# 卷积缩小2倍
def downsample(filters, size, apply_batchnorm=True):
    # 子模型, 进行卷积运算和缩小
    def result(x):
        # 正态分布初始化值
        initializer = tf.random_normal_initializer(0., 0.02)
        x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False)(x)
        # TODO 待写
        if apply_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x, training=True)
        # TODO 代写
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x
    return result

# 反卷积扩大2倍
def upsample(filters, size, apply_dropout=False):
    # 子模型, 进行反卷积运算和扩大
    def result(x):
        # TODO 待写
        x = tf.keras.layers.ReLU()(x)
        # 正态分布初始化
        initializer = tf.random_normal_initializer(0., 0.02)
        x = tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',
                                        kernel_initializer=initializer, use_bias=False)(x)
        # TODO 待写
        x = tf.keras.layers.BatchNormalization()(x, training=True)
        # 处理模型过拟合
        if apply_dropout:
            x = tf.keras.layers.Dropout(0.5)(x)
        return x
    return result

# 卷积缩小2倍
def downsample_gen(filters, size, apply_batchnorm=True):
    # 子模型, 进行卷积运算和缩小
    def result(x):
        # TODO 代写
        if apply_batchnorm:
            x = tf.keras.layers.LeakyReLU(0.2)(x)
        # 正态分布初始化值
        initializer = tf.random_normal_initializer(0., 0.02)
        x = tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False)(x)
        # TODO 待写
        if apply_batchnorm:
            x = tf.keras.layers.BatchNormalization()(x, training=True)
        return x
    return result