import numpy as np
import matplotlib.pyplot as plt
# 生成器预测, 并显示图片
def show_generator_images(model, inp, tar):
    # 预测输入图
    prediction = model(inp) # training=True
    plt.figure(figsize=(15,15))
    # 显示图
    display_list = [inp[0], tar[0], prediction[0]]
    # 标题
    title = ['Input Image', 'Target Image', 'Predicted Image']
    # 循环3张, 分别为：输入, 原图, 预测
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        if display_list[i].shape[2] == 1:
            plt.imshow(np.squeeze(display_list[i]) * 0.5 + 0.5, cmap=plt.cm.gray)
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.imsave('./tmp/prediction.jpg')
    
# 生成器预测, 并显示图片
def show_images(model, inp, tar):
    # 预测输入图
    prediction = model(inp) # training=True
    for j in range(inp.shape[0]):
        plt.figure(figsize=(15,15))
        # 显示图
        display_list = [inp[j], tar[j], prediction[j]]
        # 标题
        title = ['Input Image', 'Target Image', 'Predicted Image']
        # 循环3张, 分别为：输入, 原图, 预测
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            if display_list[i].shape[2] == 1:
                plt.imshow(np.squeeze(display_list[i]) * 0.5 + 0.5, cmap=plt.cm.gray)
            else:
                plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
