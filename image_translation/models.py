from django.db import models

# Create your models here.
from django.db import models
from django.utils import timezone

'''
    引擎信息类, 存储引擎运行对应的信息
'''
class MEngine(models.Model):
    is_start = models.BooleanField(default=False)
    name = models.CharField(max_length=200, null=True)
    Lambda = models.FloatField(null=True)
    gtc = models.BooleanField(null=True)
    path = models.CharField(max_length=200, null=True)
    create_time = models.DateTimeField(default=timezone.now)
    msg = models.CharField(max_length=200, null=True)

    def __str__(self):
        return str(self.id) + ', ' + self.name + ', ' + str(self.Lambda) \
               + ', ' + str(self.gtc) + ', ' + str(self.path)

    class Meta:
        db_table = 'mengine'


'''训练信息类, 存储每一次训练对应的信息'''
class Train(models.Model):
    name = models.CharField(max_length=50, null=True)
    input_path = models.CharField(max_length=200, null=True)
    is_pix = models.BooleanField(null=True)
    pix_base_path = models.CharField(max_length=200, null=True)
    pix_img_path = models.CharField(max_length=200, null=True)
    pix_pre_path = models.CharField(max_length=200, null=True)
    save_path = models.CharField(max_length=200, null=True)
    tmp_path = models.CharField(max_length=200, null=True)
    space = models.IntegerField(null=True)
    epochs = models.IntegerField(null=True)
    batch_size = models.IntegerField(null=True)
    buffer_size = models.IntegerField(null=True)
    is_clip = models.BooleanField(null=True)
    width = models.IntegerField(null=True)
    height = models.IntegerField(null=True)
    # 下面是逻辑要赋值的
    is_gtc = models.BooleanField(null=True)
    cur_epoch = models.IntegerField(null=True)
    tmp_epoch = models.IntegerField(null=True)
    pix_acc = models.FloatField(null=True)
    epoch_time = models.FloatField(null=True)
    total_time = models.FloatField(null=True)
    is_train = False
    create_time = models.DateTimeField(default=timezone.now)

    mengine = models.ForeignKey('MEngine', on_delete=models.CASCADE)

    def __str__(self):
        return str(self.name) + ', ' + str(self.input_path) + ', ' + str(self.is_pix) + ', ' + str(self.pix_img_path) + ', ' + \
            str(self.save_path) + ', ' + str(self.tmp_path) + ', ' + str(self.space) + ', ' + \
            str(self.epochs) + ', ' + str(self.batch_size) + ', ' + str(self.buffer_size) + ', ' + \
            str(self.is_clip) + ', ' + str(self.width) + ', ' + str(self.height) + ', ' + \
            str(self.is_gtc) + ', ' + str(self.cur_epoch) + ', ' + str(self.tmp_epoch) + ', ' + \
            str(self.pix_acc) + ', ' + str(self.epoch_time) + ', ' + str(self.is_train) + ',' + str(self.pix_pre_path)

    class Meta:
        db_table = 'train'

'''
    测试信息类
'''
class Test(models.Model):
    data_path = models.CharField(max_length=200)
    mengine = models.ForeignKey('MEngine', on_delete=models.CASCADE)

    class Meta:
        db_table = 'test'
