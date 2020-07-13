import os
import shutil
import datetime
import time
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from .Generator import Generator
from .Discriminator import Discriminator
from .ShowImage import show_generator_images
from .LoadData import LoadData
from .tfimg import *

from image_translation.models import Train
# from IPython import display

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from image_translation import views
import json

# 修改后的pix2pix模型
class Pix2PixZ():

    # 初始化模型
    def __init__(self, Lambda=100.0, input_channel=1):
        # L1的超参数
        self.Lambda = Lambda
        # 创建生成器
        self.generator = Generator(input_channel)
        # 创建鉴别器
        self.discriminator = Discriminator(input_channel)
        # 生成器的优化器
        self.adam = optimizers.Adam(2e-4, beta_1=0.5)
        # 鉴别器的优化器
        self.sgd = optimizers.Adam(2e-4, beta_1=0.5)
        # 损失函数
        self.loss = losses.BinaryCrossentropy(from_logits=False)
        # 检查点创建
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.adam, discriminator_optimizer=self.sgd,
                                             generator=self.generator, discriminator=self.discriminator)
        # 损失日志
        self.summary_writer = None
        # 通道数
        self.input_channel = input_channel

    # 每一步的训练
    '''
        训练方法, 用于训练模型
        @param inp: 灰度图
        @param tar: 真实图
    '''
    @tf.function
    def train_step(self, inp, tar, epoch):
        # 分别创建生成器和鉴别器的梯度带
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # 灰度图通过生成器 生成 假彩图
            gen_output = self.generator(inp, training = True) # training = True

            # 鉴别器将 inp和tar 这一对真实的数据进行捆绑, 并且鉴别. 我们希望这个越接近1越好
            disc_real_output = self.discriminator([inp, tar], training = True) # training = True
            # 鉴别器将 inp和假数据 进行鉴别, 我们希望这个越接近0越好
            disc_generated_output = self.discriminator([inp, gen_output], training = True) # training = True

            # 生成器损失计算
            total_gan_loss, gen_gan_loss, gen_l1_loss = self.gen_loss(disc_generated_output, gen_output, tar)
            # 鉴别器损失计算
            disc_loss = self.disc_loss(disc_real_output, disc_generated_output)

        # 生成器梯度下降
        generator_gradients = gen_tape.gradient(total_gan_loss, self.generator.trainable_variables)
        self.adam.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        # 鉴别器梯度下降
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.sgd.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return total_gan_loss, gen_gan_loss, gen_l1_loss, disc_loss

    # 训练模型
    def fit(self, train):
        # dataset
        l = LoadData(width=train.width, height=train.height, buffer_size=train.buffer_size,
                     batch_size=train.batch_size, is_gtc=train.is_gtc, is_clip=train.is_clip)
        train_ds = l.load_data(train.input_path)
        tmp_epoch = train.tmp_epoch
        epochs = train.epochs
        checkpoint_dir = train.tmp_path
        space = train.space
        # 由于异常终止, 所以需要从中途开始再次训练
        start_epoch = tmp_epoch
        # 更新cur_epoch
        train.cur_epoch = start_epoch
        train.save()

        # 日志和保存路径
        self.summary_writer = tf.summary.create_file_writer(os.path.join(checkpoint_dir, 'logs/') + datetime.datetime.now().strftime("%Y-%m-%d x %H.%M.%S"))
        checkpoint_prefix = os.path.join(os.path.join(checkpoint_dir, 'checkpoints/'), "ckpt")
        # genloss、discloss、pixacc、time、total_time
        if not os.path.exists(os.path.join(checkpoint_dir, 'datas.json')):
            with open(os.path.join(checkpoint_dir, 'datas.json'), 'w', encoding='utf-8') as fdatas:
                datas = {'total_gan_loss': [], 'gen_gan_loss': [], 'gen_l1_loss': [], 'disc_loss': [], 'pixacc': [], 'time': [], 'total_time': 0.0}
                json.dump(datas, fdatas, indent=4, ensure_ascii=True)
        with open(os.path.join(checkpoint_dir, 'datas.json'), 'r', encoding='utf-8') as fdatas:
            datas = json.load(fdatas)
        # 切片到start_epoch
        datas['total_gan_loss'] = datas['total_gan_loss'][:start_epoch]
        datas['gen_gan_loss'] = datas['gen_gan_loss'][:start_epoch]
        datas['gen_l1_loss'] = datas['gen_l1_loss'][:start_epoch]
        datas['disc_loss'] = datas['disc_loss'][:start_epoch]
        datas['pixacc'] = datas['pixacc'][:start_epoch]
        datas['time'] = datas['time'][:start_epoch]
        datas['total_time'] = 0.0
        for i in range(start_epoch):
            datas['total_time'] += datas['time'][i]
        # 更新时间
        train.total_time = datas['total_time']
        train.save()

        # 迭代循环
        for epoch in range(start_epoch, epochs):
            # 记录开始的时间
            start = time.time()
            # 清除输出
            # display.clear_output(wait=True)
            # for example_input, example_target in test_ds.take(1):
            #     show_generator_images(self.generator, example_input, example_target)
            if train.is_pix:
                train.pix_acc = self.pix_acc(train.pix_base_path, train.pix_img_path, train.pix_pre_path)
                datas['pixacc'].append(train.pix_acc)
                train.save()
                with open(os.path.join(checkpoint_dir, 'datas.json'), 'w', encoding='utf-8') as fdatas:
                    json.dump(datas, fdatas, indent=4, ensure_ascii=True)

            train.cur_epoch = epoch
            train.save()
            total_gan_loss = gen_gan_loss = gen_l1_loss = disc_loss = .0
            # 训练
            for inp, tar in train_ds:
                # 训练每个批次
                tgl, ggl, gll, dl = self.train_step(inp, tar, epoch)
                total_gan_loss += tgl
                gen_gan_loss += ggl
                gen_l1_loss += gll
                disc_loss += dl
                # 是否结束
                flag = views.is_trains.get(str(train.id))
                if (not (flag is None)) and (not flag):
                    epoch += 1
                    train.epochs = train.cur_epoch = epoch
                    t = time.time()-start
                    # 花费时间
                    train.epoch_time = t
                    train.total_time += t
                    datas['time'].append(t)
                    datas['total_time'] += t
                    train.save()
                    with open(os.path.join(checkpoint_dir, 'datas.json'), 'w', encoding='utf-8') as fdatas:
                        json.dump(datas, fdatas, indent=4, ensure_ascii=True)
                    # 最后训练结束, 保存检查点
                    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoints')):
                        shutil.rmtree(os.path.join(checkpoint_dir, 'checkpoints'))
                    self.checkpoint.save(file_prefix=checkpoint_prefix)
                    return
            # 将损失记录到TensorBoard
            with self.summary_writer.as_default():
                tf.summary.scalar('total_gan_loss', total_gan_loss, step=epoch)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)
            datas['total_gan_loss'].append(float(total_gan_loss.numpy()))
            datas['gen_gan_loss'].append(float(gen_gan_loss.numpy()))
            datas['gen_l1_loss'].append(float(gen_l1_loss.numpy()))
            datas['disc_loss'].append(float(disc_loss.numpy()))
            # 保存检查点
            if (epoch + 1) % space == 0:
                if os.path.exists(os.path.join(checkpoint_dir, 'checkpoints')):
                    shutil.rmtree(os.path.join(checkpoint_dir, 'checkpoints'))
                self.checkpoint.save(file_prefix=checkpoint_prefix)
                train.tmp_epoch = epoch
            t = time.time()-start
            # 花费时间
            train.epoch_time = t
            train.total_time += t
            datas['time'].append(t)
            datas['total_time'] += t
            with open(os.path.join(checkpoint_dir, 'datas.json'), 'w', encoding='utf-8') as fdatas:
                json.dump(datas, fdatas, indent=4, ensure_ascii=True)
            train.save()

        train.tmp_epoch = train.cur_epoch = epochs
        train.save()
        # 最后训练结束, 保存检查点
        if os.path.exists(os.path.join(checkpoint_dir, 'checkpoints')):
            shutil.rmtree(os.path.join(checkpoint_dir, 'checkpoints'))
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    # 生成器损失函数
    def gen_loss(self, disc_gen_out, gen_out, target):
        # 生成器希望：生成的图像被鉴别器检测时, 希望接近1
        gen_gan_loss = self.loss(tf.ones_like(disc_gen_out), disc_gen_out)
        # gen_gan_loss = tf.reduce_mean(-tf.math.log(disc_gen_out))
        # 生成图像和真实图像的差距
        gen_l1_loss = tf.reduce_mean(tf.abs(gen_out - target))
        return gen_gan_loss + (self.Lambda * gen_l1_loss), gen_gan_loss, gen_l1_loss

    # 鉴别器损失函数
    def disc_loss(self, disc_real_out, disc_gen_out):
        # 鉴别器希望：鉴别真实的图片接近1
        real_loss = self.loss( tf.ones_like(disc_real_out), disc_real_out)
        # 鉴别器希望：鉴别伪造的图片接近0
        gen_loss = self.loss(tf.zeros_like(disc_gen_out), disc_gen_out)
        return real_loss + gen_loss

    # 保存模型
    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        filenames = os.listdir(path)
        for filename in filenames:
            if (filename == 'checkpoint') or ('ckpt' in filename):
                os.remove(os.path.join(path, filename))
        checkpoint_prefix = os.path.join(path, "ckpt")
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    # 恢复模型
    def load(self, path):
        self.checkpoint.restore(tf.train.latest_checkpoint(path))

    # 预测图片
    def predict(self, img_path):
        # 读取图片
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        if self.input_channel == 1:
            img = tf.cast(img, tf.float32)/255.0
            lab = rgb2lab(img)
            l, a, b = preprocess_lab(lab)
            img = tf.expand_dims(l, axis=2)
        else:
            img = (tf.cast(img, tf.float32) / 127.5) - 1

        # 规则化的计算
        shape = img.shape
        target_height = tf.cast(tf.math.ceil(shape[0] / 256.0) * 256, tf.int32)
        target_width = tf.cast(tf.math.ceil(shape[1] / 256.0) * 256, tf.int32)
        offset_height = tf.cast((target_height - shape[0]) / 2, tf.int32)
        offset_width = tf.cast((target_width - shape[1]) / 2, tf.int32)
        # 进行规则化
        img = tf.image.pad_to_bounding_box(img, offset_height, offset_width, target_height, target_width)
        # 预测
        pre = self.generator(tf.expand_dims(img, axis=0))[0]
        # 反规则化
        img = tf.image.crop_to_bounding_box(img, offset_height, offset_width, shape[0], shape[1])
        pre = tf.image.crop_to_bounding_box(pre, offset_height, offset_width, shape[0], shape[1])
        if self.input_channel == 1:
            l = tf.unstack(img, axis=2)[0]
            a, b = tf.unstack(pre, axis=2)
            lab = deprocess_lab(l, a, b)
            pre = lab2rgb(lab)
            pre = tf.cast(pre*255, tf.uint8)
        else:
            # 从[-1, 1]到[0, 255]
            pre = tf.cast((pre + 1.0) * 127.5, tf.uint8)
        # 生成预测图片的全路径
        pre_path = img_path[:img_path.rfind('.')] + '-pre.jpg'
        # 保存图片
        with tf.io.gfile.GFile(pre_path, 'wb') as f:
            f.write(tf.image.encode_jpeg(pre, quality=100).numpy())

    # 像素准确度
    def pix_acc(self, pix_base_path, pix_img_path, pix_pre_path):
        # 读取图片
        img = tf.io.read_file(os.path.join(pix_base_path, pix_img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        if self.input_channel == 1:
            img = tf.cast(img, tf.float32)/255.0
            lab = rgb2lab(img)
            l, a, b = preprocess_lab(lab)
            inp = tf.expand_dims(l, axis=2)
            tar = tf.stack([a, b], axis=2)
        else:
            w = tf.shape(img)[1] // 2
            inp = img[:, :w, :]
            tar = img[:, w:, :]
            inp = (tf.cast(inp, tf.float32) / 127.5) - 1
            tar = (tf.cast(tar, tf.float32) / 127.5) - 1

        shape = inp.shape
        target_height = tf.cast(tf.math.ceil(shape[0] / 256.0) * 256, tf.int32)
        target_width = tf.cast(tf.math.ceil(shape[1] / 256.0) * 256, tf.int32)
        offset_height = tf.cast((target_height - shape[0]) / 2, tf.int32)
        offset_width = tf.cast((target_width - shape[1]) / 2, tf.int32)
        inp = tf.image.pad_to_bounding_box(inp, offset_height, offset_width, target_height,
                                           target_width)
        pre = self.generator(tf.expand_dims(inp, axis=0))[0]
        pre = tf.image.crop_to_bounding_box(pre, offset_height, offset_width, shape[0],
                                            shape[1])
        inp = tf.image.crop_to_bounding_box(inp, offset_height, offset_width, shape[0],
                                            shape[1])
        if self.input_channel == 1:
            l = tf.unstack(inp, axis=2)[0]
            a, b = tf.unstack(pre, axis=2)
            lab = deprocess_lab(l, a, b)
            pre = lab2rgb(lab)
            pre = tf.cast(pre*255, tf.int32).numpy()
            tar = tf.cast(img*255, tf.int32).numpy()
        else:
            pre = tf.cast((pre + 1.0) * 127.5, tf.int32).numpy()
            tar = tf.cast((tar + 1.0) * 127.5, tf.int32).numpy()

        shape = tar.shape
        pix_acc = np.sum(pre == tar) / (shape[0] * shape[1] * shape[2])
        # 预测路径
        res_pre_path = os.path.join(pix_base_path, pix_pre_path)
        # 保存图片
        with tf.io.gfile.GFile(res_pre_path, 'wb') as f:
            f.write(tf.image.encode_jpeg(pre, quality=100).numpy())
        return pix_acc
