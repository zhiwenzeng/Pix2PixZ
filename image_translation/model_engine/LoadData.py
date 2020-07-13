import tensorflow as tf
from .tfimg import *


class LoadData(object):

    def __init__(self, width=256, height=256, buffer_size=1000, batch_size=50,
                 is_gtc=True, is_clip=True):
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.load_data = None
        # 这里就需要判断是否为gtc以及是否clip
        if is_gtc:
            if is_clip:
                self.load_data = self.load_clip_gtc
            else:
                self.load_data = self.load_resize_gtc
        else:
            if is_clip:
                self.load_data = self.load_clip
            else:
                self.load_data = self.load_resize

    # gtc翻译的数据, 并且是resize
    def load_resize_gtc(self, input_path):
        data = tf.data.Dataset.list_files(input_path + r'/*.jpg')
        data = data.map(self.preprocess_img_load_resize_gtc).shuffle(self.buffer_size).batch(self.batch_size)
        return data

    @tf.function
    def preprocess_img_load_resize_gtc(self, filename):
        # 加载图片
        img = self.load_img(filename)
        # 重置大小
        img = self.resize_img(img)
        # 随机翻转
        img = tf.image.flip_left_right(img)
        # 标准化
        img = tf.cast(img, tf.float32)/255.0
        lab = rgb2lab(img)
        l, a, b = preprocess_lab(lab)
        input_image = tf.expand_dims(l, axis=2)
        target_image = tf.stack([a, b], axis=2)
        return input_image, target_image

    # gtc翻译的数据, 并且是clip
    def load_clip_gtc(self, input_path):
        data = tf.data.Dataset.list_files(input_path + r'/*.jpg')
        data = data.map(self.preprocess_img_load_clip_gtc).shuffle(self.buffer_size).batch(self.batch_size)
        return data

    @tf.function
    def preprocess_img_load_clip_gtc(self, filename):
        # 加载图片
        img = self.load_img(filename)
        # 随机裁剪
        img = self.random_crop_gtc(img)
        # 随机翻转
        img = tf.image.flip_left_right(img)
        # 标准化
        img = tf.cast(img, tf.float32)/255.0
        lab = rgb2lab(img)
        l, a, b = preprocess_lab(lab)
        input_image = tf.expand_dims(l, axis=2)
        target_image = tf.stack([a, b], axis=2)
        return input_image, target_image

    # 普通翻译的数据, 并且是resize
    def load_resize(self, input_path):
        data = tf.data.Dataset.list_files(input_path + r'/*.jpg')
        data = data.map(self.preprocess_img_load_resize).shuffle(self.buffer_size).batch(self.batch_size)
        return data

    @tf.function
    def preprocess_img_load_resize(self, filename):
        # 加载图片
        img = self.load_img(filename)
        # 剪切成一半
        input_image, target_image = self.cut_img(img)
        # 重置大小
        input_image = self.resize_img(input_image)
        target_image = self.resize_img(target_image)
        # 随机翻转
        input_image, target_image = self.random_flip(input_image, target_image)
        # 标准化
        input_image = self.normalize(input_image)
        target_image = self.normalize(target_image)
        return input_image, target_image

    # 普通翻译的数据, 并且是clip
    def load_clip(self, input_path):
        data = tf.data.Dataset.list_files(input_path + r'/*.jpg')
        data = data.map(self.preprocess_img_load_clip).shuffle(self.buffer_size).batch(self.batch_size)
        return data

    @tf.function
    def preprocess_img_load_clip(self, filename):
        # 加载图片
        img = self.load_img(filename)
        # 剪切成一半
        input_image, target_image = self.cut_img(img)
        # 随机裁剪
        input_image, target_image = self.random_crop(input_image, target_image)
        # 随机翻转
        input_image, target_image = self.random_flip(input_image, target_image)
        # 标准化
        input_image = self.normalize(input_image)
        target_image = self.normalize(target_image)
        return input_image, target_image

    # 加载图片
    def load_img(self, filename):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    # 减成一半
    def cut_img(self, img):
        w = tf.shape(img)[1] // 2
        input_img = img[:, :w, :]
        target_img = img[:, w:, :]
        return input_img, target_img

    # 重置大小
    def resize_img(self, img):
        img = tf.image.resize(img, [self.height, self.width], method=tf.image.ResizeMethod.BILINEAR)
        return img

    # gtc随机裁剪
    def random_crop_gtc(self, img):
        img = tf.image.random_crop(img, size=[self.height, self.width, 3])
        return img

    # 随机裁剪
    def random_crop(self, input_image, target_image):
        images = tf.stack([input_image, target_image], axis=0)
        res = tf.image.random_crop(images, size=[2, self.height, self.width, 3])
        return res[0], res[1]

    # 随机翻转
    def random_flip(self, input_image, target_image):
        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            target_image = tf.image.flip_left_right(target_image)
        return input_image, target_image

    # 标准化到(-1, 1)
    def normalize(self, x):
        return (tf.cast(x, tf.float32) / 127.5) - 1
