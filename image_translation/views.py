import os
import time
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
from image_translation.engineManage import Engine, EngineManage
from image_translation.models import MEngine, Train
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from django.views.decorators.csrf import csrf_exempt,csrf_protect
import uuid
import shutil
import base64


# 这个是引擎测试页的内容
indexs = os.path.join(os.getcwd(), r'static/indexs/')
# 是否开启训练过
is_trains = {}
# 文件上传图片的路径
upload = os.path.abspath('static/upload/')
# pix预测图片的路径
imgs_path = os.path.abspath('static/imgs/')


# 首页需要有现在目前已经启动的 翻译应用
def index(request):
    content = {}
    # 需要查找出对应的应用名和url地址和id
    # 只要去管理器那边查找即可
    data = EngineManage().find_all()
    content['data'] = data
    return render(request, 'index.html', content)


# 翻译应用的创建首页
def translation_index(request):
    return render(request, 'translation/index.html', {})


# 创建引擎的接口
def create_engine(request):
    result = {}
    name = request.GET.get('name')
    Lambda = request.GET.get('lambda')
    gtc = (request.GET.get('gtc') == 'YES')
    msg = request.GET.get('msg')
    # 将创建的引擎信息存入数据库
    MEngine.objects.create(name=name, Lambda=Lambda, gtc=gtc, msg=msg)
    return JsonResponse(result)


# 引擎列表接口
def engine_list(request, page_num):
    MEngine.objects.all().update(is_start=False)
    engines = EngineManage().find_all()
    for key in engines.keys():
        MEngine.objects.filter(id=key).update(is_start=True)

    content = {}
    mengine_all = MEngine.objects.order_by('-is_start', 'create_time').all()
    paginator = Paginator(mengine_all, 4)
    try:
        mengine_ds = paginator.page(page_num)
    except PageNotAnInteger:
        mengine_ds = paginator.page(1)
    except EmptyPage:
        mengine_ds = paginator.page(paginator.num_pages)
    '''
    trains = []
    mengines = []
    for mengine in mengine_ds:
        mengine.is_start = not (engines.get(str(mengine.id)) is None)
        mengines.append(mengine)
        trains_ds = Train.objects.filter(mengine_id=mengine.id).order_by('-id')
        if len(trains_ds) >= 1:
            trains.append(trains_ds[0])
            if trains[-1].cur_epoch != trains[-1].epochs:
                # 如果is_trains中对应的train_id已经点击过训练, 则说明正在训练中
                trains[-1].is_train = not (is_trains.get(str(trains[-1].id)) is None)
        else:
            trains.append(None)
    content['engines'] = zip(mengines, trains)
    '''
    content['mengines'] = mengine_ds
    content['page'] = mengine_ds
    content['paginator'] = paginator
    return render(request, 'translation/engine_list.html', content)


def engine_info(request):
    engine_id = request.GET.get('engine_id')
    mengine = MEngine.objects.filter(id=engine_id)[0]
    mengine.is_start = not EngineManage().find_all().get(engine_id) is None
    trains = Train.objects.filter(mengine_id=engine_id).order_by('-id')
    train = None
    if len(trains) != 0:
        train = trains[0]
    return render(request, 'translation/engine_info.html', {'mengine': mengine, 'train': train})


def start_engine(request):
    engine_id = request.GET.get('engine_id')
    path = EngineManage().get(engine_id).mengine.path
    content = {}
    try:
        if path is None:
            content['status'] = 'success'
            return JsonResponse(content)
        if not os.path.exists(path):
            content['status'] = 'error'
            content['info'] = '{}路径不存在'.format(path)
        elif not ('checkpoint' in os.listdir(path)):
            content['status'] = 'error'
            content['info'] = '{}不存在ckpt文件'.format(path)
        else:
            EngineManage().load_weight(engine_id, path)
            content['status'] = 'success'
            content['info'] = path
    except Exception as e:
        content['status'] = 'exception'
        content['error'] = e
    if content['status'] != 'success':
        MEngine.objects.filter(id=engine_id).update(path=None)
    return JsonResponse(content)


# 引擎删除接口
def delete_engine(request, id):
    mengine = MEngine.objects.filter(id=id)[0]
    mengine.delete()
    content = {'status': 'success', 'info': '引擎删除成功'}
    return JsonResponse(content)


# 为引擎添加训练
def create_train(request):
    # 获取引擎id
    engine_id = request.GET.get('engine_id')
    # 获取当前引擎的最后一次训练
    trains = Train.objects.filter(mengine_id=engine_id).order_by('-id')
    last_train = None
    if len(trains) != 0:
        last_train = trains[0]
    # 判断最后一次训练是否完成
    if not (last_train is None) and last_train.cur_epoch != last_train.epochs:
        return JsonResponse({'status': 'error'})

    name = request.GET.get('name')
    if name == '':
        name = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    input_path = request.GET.get('input_path')
    input_path = os.path.abspath(input_path)
    is_pix = request.GET.get('is_pix') == 'true'
    if is_pix:
        pix_img_path = request.GET.get('pix_img_path')
        pix_img_path = os.path.abspath(pix_img_path)
    else:
        pix_img_path = None
    save_path = request.GET.get('save_path')
    save_path = os.path.abspath(save_path)
    tmp_path = request.GET.get('tmp_path')
    tmp_path = os.path.abspath(tmp_path)
    space = request.GET.get('space')
    epochs = request.GET.get('epochs')
    batch_size = request.GET.get('batch_size')
    buffer_size = request.GET.get('buffer_size')
    is_clip = request.GET.get('is_clip') == 'true'
    width = request.GET.get('width')
    height = request.GET.get('height')

    is_gtc = MEngine.objects.filter(id=engine_id)[0].gtc
    cur_epoch = 0
    tmp_epoch = 0
    pix_acc = 0.0
    epoch_time = 0.0
    total_time = 0.0

    if not (pix_img_path is None):
        base_path = pix_img_path[: pix_img_path.rfind('\\')]
        # 图片名字
        img_name = pix_img_path[pix_img_path.rfind('\\')+1: ]
        # 预测名字
        pre_name = img_name[: img_name.rfind('.')] + '-pre.jpg'
        # 图片的复制
        shutil.copy(pix_img_path, os.path.join(imgs_path, img_name))
    else:
        img_name = None
        pre_name = None

    if request.GET.get('input_path') == '空':
        cur_epoch = epochs = -1

    Train.objects.create(name=name, input_path=input_path, is_pix=is_pix, pix_img_path=img_name, pix_pre_path=pre_name,
                         save_path=save_path, tmp_path=tmp_path, space=space, epochs=epochs, pix_base_path=base_path,
                         batch_size=batch_size, buffer_size=buffer_size, is_clip=is_clip, width=width,
                         height=height, is_gtc=is_gtc, cur_epoch=cur_epoch, tmp_epoch=tmp_epoch,
                         pix_acc=pix_acc, epoch_time=epoch_time, total_time=total_time, mengine_id=engine_id)
    content = {'status': 'success'}
    return JsonResponse(content)


def train(request):
    engine_id = request.GET.get('engine_id')
    train_id = request.GET.get('train_id')
    is_keep = request.GET.get('is_keep') == 'true'
    if is_keep:
        train = Train.objects.filter(id=train_id)[0]
        path = os.path.join(train.tmp_path, 'checkpoints')
        EngineManage().load_weight(engine_id, path)
        mengine = MEngine.objects.filter(id=engine_id)[0]
        mengine.path = path
        mengine.save()
    EngineManage().train(engine_id, train_id)
    is_trains[train_id] = True
    content = {'status': 'success'}
    return JsonResponse(content)


# 加载权重
def load_weight(request):
    # 先获取引擎的id和'加载路径'
    engine_id = request.GET.get('engine_id')
    path = request.GET.get('path')
    # 路径如果为空,则默认获取最后一次训练保存的模型路径
    if path is None:
        train = Train.objects.filter(mengine_id=engine_id).order_by('-id')[0]
        path = train.save_path
    # 将路径变为 绝对路径
    path = os.path.abspath(path)
    # 如果ckpt文件存在,说明有对应的模型
    if not os.path.exists(path):
        content = {'status': 'error', 'info': '{}路径不存在'.format(path)}
        return JsonResponse(content)
    elif 'checkpoint' in os.listdir(path):
        mengine = MEngine.objects.filter(id=engine_id)[0]
        mengine.path = path
        mengine.save()
        EngineManage().load_weight(engine_id, path)
        content = {'status': 'success', 'info': path}
        return JsonResponse(content)
    else:
        # 如果没找到,则说明不存在
        content = {'status': 'error', 'info': '{}文件夹中不存在ckpt的文件'.format(path)}
        return JsonResponse(content)


def save_weight(request):
    engine_id = request.GET.get('engine_id')
    path = request.GET.get('path')
    path = os.path.abspath(path)
    content = {}
    if not os.path.exists(path):
        content['status'] = 'error'
        content['info'] = path+'文件路径不存在'
    else:
        EngineManage().save_weight(engine_id, path)
        content['status'] = 'success'
        content['info'] = '模型参数保存成功,在'+path+'文件夹中'
    return JsonResponse(content)


def train_list(request, page_num):
    engine_id = request.GET.get('engine_id')
    content = {}
    trains = Train.objects.filter(mengine_id=engine_id).order_by('-create_time')
    paginator = Paginator(trains, 10)
    try:
        train_ds = paginator.page(page_num)
    except PageNotAnInteger:
        train_ds = paginator.page(1)
    except EmptyPage:
        train_ds = paginator.page(paginator.num_pages)

    trains = list(train_ds)
    for i in range(len(trains)):
        if is_trains.get(str(trains[i].id)) is None:
            trains[i].is_train = False
        else:
            trains[i].is_train = is_trains.get(str(trains[i].id))
        trains[i].save_path = os.path.abspath(trains[i].save_path)

    content['trains'] = trains
    content['page'] = train_ds
    content['paginator'] = paginator
    return render(request, r'translation/train_list.html', content)


# 设置刷新在本页面和路径导航
def set_content_url(request):
    # 导航的url和level以及name
    url = request.GET.get('path')
    level = int(request.GET.get('level'))
    name = request.GET.get('name')
    # 设置刷新的url
    request.session['content_url'] = url
    # 路径导航的key
    nav_paths = 'nav_paths'
    # 如果key为none
    if request.session.get(nav_paths) is None:
        request.session[nav_paths] = []
    # 切片到level
    paths = request.session[nav_paths][:level]
    # 添加本次的路径
    paths.append({'url': url, 'level': level, 'name': name})
    # 最后更新session
    request.session[nav_paths] = paths
    request.session['nav_paths_length'] = level
    # 返回json数据
    content = {'status': 'success', 'nav_paths': paths}
    return JsonResponse(content)

import json

def train_info(request):
    train_id = request.GET.get('train_id')
    train = Train.objects.filter(id=train_id)[0]
    train.is_train = not (is_trains.get(str(train.id)) is None)

    inp = tar = None
    if train.is_pix:
        inp_path = os.path.abspath(os.path.join(train.pix_base_path, train.pix_img_path))
        tar_path = os.path.abspath(os.path.join(train.pix_base_path, train.pix_pre_path))
        if os.path.exists(inp_path):
            with os.path.exists(inp_path) and open(inp_path, 'rb') as f:
                inp = 'data:image/png;base64,'+base64.b64encode(f.read()).decode()
            if not os.path.exists(tar_path):
                EngineManage().pix_acc(train)

        if os.path.exists(tar_path):
            with open(tar_path, 'rb') as f:
                tar = 'data:image/png;base64,'+base64.b64encode(f.read()).decode()

    with open(os.path.join(train.tmp_path, 'datas.json'), 'r', encoding='utf-8') as fdatas:
        datas = json.load(fdatas)
        datas['epochs'] = [i for i in range(len(datas['pixacc']))]
        content = {'train': train, 'datas': datas, 'inp': inp, 'tar': tar}
    return render(request, 'translation/train_info.html', content)


def create_train_page(request):
    content = {'engine_id': None, 'engine_name': None, 'engine_gtc': None}
    for key in content:
        content[key] = request.GET.get(key)
    return render(request, 'translation/create_train.html', content)


# 删除未完成的训练
def delete_train(request):
    train_id = request.GET.get('train_id')
    train = Train.objects.filter(id=train_id)[0]
    train.delete()
    MEngine.objects.filter(id=train.mengine_id).update(path=None)
    content = {'status': 'success', 'info': '删除成功'}
    return JsonResponse(content)


def train_stop(request):
    train_id = request.GET.get('train_id')
    is_trains[train_id] = False
    return JsonResponse({'status': 'success'})


@csrf_exempt
def predict(request):
    # 获取引擎id
    engine_id = request.POST.get('engine_id')
    # 上传图片
    img = request.FILES.get('img')
    img_name = str(uuid.uuid4())+'.jpg'
    file_path = os.path.abspath(os.path.join(upload, img_name))
    with open(file_path, 'wb') as f:
        for chunk in img.chunks():
            f.write(chunk)
    # 预测图片
    EngineManage().predict(engine_id, file_path)
    predict_path = img_name[: img_name.rfind('.')]+'-pre.jpg'
    with open(os.path.abspath(os.path.join(upload, img_name)), 'rb') as f:
        img = base64.b64encode(f.read()).decode()
    with open(os.path.abspath(os.path.join(upload, predict_path)), 'rb') as f:
        predict = base64.b64encode(f.read()).decode()
    return JsonResponse({'status': 'success', 'img': 'data:image/png;base64,'+img,
                         'predict': 'data:image/png;base64,'+predict})

