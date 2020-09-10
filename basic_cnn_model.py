# 导入需要的包
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 创建一个 cnn 模型
def define_cnn_model():
    model = Sequential()
    # 卷积层     
    model.add(Conv2D(32, (3, 3),
                     activation='relu',
                     kernel_initializer='he_uniform',
                     padding='same',
                     input_shape=(200, 200, 3)))
    # 最大池化层    
    model.add(MaxPooling2D((2, 2)))
    # Flatten 层
    model.add(Flatten())
    # 全连接层
    model.add(Dense(128,
                    activation='relu',
                    kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_cnn_model():
    # 实例化模型
    model = define_cnn_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory(
        './datasets/ma1ogo3ushu4ju4ji2-momodel/dogs_cats/data/train/',
        class_mode='binary',
        batch_size=64,
        target_size=(200, 200))
    # 训练模型
    model.fit_generator(train_it,
                        steps_per_epoch=len(train_it),
                        epochs=20,
                        verbose=1)
    # 把模型保存到 results 文件夹
    model.save("./results/basic_cnn_model.h5")


if __name__ == "__main__":
    train_cnn_model()
