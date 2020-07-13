# 导入图像翻译模型
from .Pix2PixZ import Pix2PixZ
from image_translation.models import MEngine


class REngine(object):

    # 初始化引擎
    '''
         @:param Lamdba 超参数
    '''
    def __init__(self, Lambda, input_channel):
        self.it = Pix2PixZ(Lambda, input_channel)

    # 训练模型
    '''@:param train 训练信息'''
    def train(self, train):
        self.it.fit(train)
        self.save(train.save_path)
        MEngine.objects.filter(id=train.mengine_id).update(path=train.save_path)

    # 保存模型
    '''
        @:param save_path 保存模型
    '''
    def save(self, save_path):
        self.it.save(save_path)

    # 加载模型
    '''
        @:param load_path 加载模型
    '''
    def load(self, load_path):
        self.it.load(load_path)

    # 预测
    def predict(self, img_path):
        return self.it.predict(img_path)

    # 准确度计算
    def pix_acc(self, train):
        return self.it.pix_acc(train.pix_base_path, train.pix_img_path, train.pix_pre_path)
