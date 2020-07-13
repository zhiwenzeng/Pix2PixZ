'''
    图像翻译的url转发子模块
'''
from django.urls import path
from image_translation import views

urlpatterns = [
    path('', views.translation_index),
    path('create_engine/', views.create_engine),
    path('engine_list/<int:page_num>', views.engine_list),
    path('engine_info', views.engine_info),
    path('start_engine', views.start_engine),
    path('delete_engine/<int:id>/', views.delete_engine),
    path('create_train_page', views.create_train_page),
    path('create_train', views.create_train),
    path('train', views.train),
    path('load_weight', views.load_weight),
    path('save_weight', views.save_weight),
    path('train_list/<int:page_num>', views.train_list),
    path('set_content_url', views.set_content_url),
    path('train_info', views.train_info),
    path('delete_train', views.delete_train),
    path('train_stop', views.train_stop),
    path('predict', views.predict),
]
