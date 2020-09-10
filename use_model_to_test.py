from keras.models import load_model
# 修改 model_path 为你自己保存的模型的位置
model_path = './datasets/ma1ogo3ushu4ju4ji2-momodel/dogs_cats/model/basic_cnn_model.h5'
# 载入模型
model = load_model(model_path)

import os, random
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
%matplotlib inline

def read_random_image():
    folder=r"./datasets/ma1ogo3ushu4ju4ji2-momodel/dogs_cats/data/test/"
    file_path = folder + random.choice(os.listdir(folder))
    pil_im = Image.open(file_path, 'r')
    return pil_im
    

def get_predict(pil_im, model):
    # 对图片进行缩放 
    pil_im = pil_im.resize((200, 200))
    # 将格式转为 numpy array 格式
    array_im = np.asarray(pil_im)
    # 对图片进行预测
    result = model.predict([[array_im]])
    if result[0][0] > 0.5:
        print("预测结果是：狗")
    else:
        print("预测结果是：猫")
        
pil_im = read_random_image()
imshow(np.asarray(pil_im))
get_predict(pil_im, model)