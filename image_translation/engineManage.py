import threading
from image_translation.model_engine.REngine import REngine
from image_translation.models import MEngine, Train

'''
    存储r_engine和m_engine
'''
class Engine(object):

    def __init__(self, mengine, rengine):
        self.mengine = mengine
        self.rengine = rengine

    

'''
    引擎启动管理器，单利模式
'''
class EngineManage(object):
    _instance_lock = threading.Lock()
    _is_init = True

    def __init__(self, *args, **kwargs):
        if EngineManage._is_init:
            EngineManage._is_init = False
            self.engines = {}

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with EngineManage._instance_lock:
                if not hasattr(cls, '_instance'):
                    EngineManage._instance = super().__new__(cls)
        return EngineManage._instance

    '''遍历已经启动的id和engine'''
    def find_all(self):
        return self.engines

    '''通过id加载引擎'''
    def load(self, id):
        # 获取 引擎信息
        mengine = MEngine.objects.filter(id=id)[0]
        rengine = REngine(mengine.Lambda, 1 if mengine.gtc else 3)
        self.engines[id] = Engine(mengine, rengine)

    '''通过id获取引擎'''
    def get(self, id):
        id = str(id)
        if self.engines.get(id) is None:
            self.load(id)
        else:
            mengine = MEngine.objects.filter(id=id)[0]
            self.engines[id].mengine = mengine
        return self.engines[id]

    '''使用数据训练引擎'''
    def train(self, engine_id, train_id):
        engine = self.get(engine_id)
        rengine = engine.rengine
        train = Train.objects.filter(id=train_id)[0]
        threading.Thread(target=rengine.train, args=(train, )).start()

    '''加载数据'''
    def load_weight(self, engine_id, path):
        engine = self.get(engine_id)
        rengine = engine.rengine
        rengine.load(path)

    '''保存数据'''
    def save_weight(self, engine_id, path):
        rengine = self.get(engine_id).rengine
        rengine.save(path)

    '''预测模块'''
    def predict(self, engine_id, img_path):
        rengine = self.get(engine_id).rengine
        rengine.predict(img_path)

    def pix_acc(self, train):
        rengine = self.get(train.mengine_id).rengine
        return rengine.pix_acc(train)
