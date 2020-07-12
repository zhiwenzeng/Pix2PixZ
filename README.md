# Pix2PixZ
2020年本科生zzw的毕业设计，基于tensorflow2.x框架搭建的pix2pix深度网络来完成图像翻译任务，并且配合Django来实现可视化操作。
## 软件架构
pix2pix深度网络模型是基于Tensorflow2.x所搭建，使用Train和Engine两个自定义工具来驱动pix2pix深度网络模型，通过一个EngineManage来管理Engine，从而使上层能够方便调用。
## 安装教程
1、安装anaconda环境<br/>
2、克隆本仓库<br/>
3、导入运行环境 conda env create -f Pix2PixZ.yaml
## 使用说明
1、进入项目跟目录，输入python manage.py runserver<br/>
2、首页拥有使用说明
