# 导入需要的包
import os
from shutil import copy2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def define_transfer_model():
    # 构建不带分类器的预训练模型
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)

    # 添加一个分类器，假设我们有200个类
    prediction = Dense(1, activation='sigmoid')(x)

    # 首先，我们只训练顶部的几层（随机初始化的层）
    # 锁住所有 InceptionV3 的卷积层
    for layer in base_model.layers:
        layer.trainable = False
    model = Model(inputs=base_model.input, outputs=prediction)
    # 编译模型
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_transfer_model():
    # 实例化模型
    model = define_transfer_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory(
        './datasets/ma1ogo3ushu4ju4ji2-momodel/dogs_cats/data/train/',
        class_mode='binary',
        batch_size=64,
        target_size=(224, 224))
    # 训练模型
    model.fit_generator(train_it,
                        steps_per_epoch=len(train_it),
                        epochs=5,
                        verbose=1)


if __name__ == "__main__":
    # 把 InceptionV3 的预训练模型拷贝到 .keras 文件夹下，避免重复下载
    if not os.path.exists('/home/jovyan/.keras/models/'):
        os.mkdir('/home/jovyan/.keras/models')
    copy2("./datasets/ma1ogo3ushu4ju4ji2-momodel/dogs_cats/model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
          "/home/jovyan/.keras/models")
    # 开始训练模型
    train_transfer_model()
