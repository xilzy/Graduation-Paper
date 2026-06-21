import os
import glob
import numpy as np
import h5py
import tensorflow.compat.v1 as tf

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 32, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 128, "The size of patch to use [33]")
flags.DEFINE_integer("label_size", 128, "The size of label to produce [21]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]") #尺度
flags.DEFINE_integer("stride", 24, "The size of stride to apply input image [14]")     #步长
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")  #控制训练还是测试(这里恒置为True即可)
FLAGS = flags.FLAGS


# 获得每一张图像的绝对路径
def make_data(data, data_dir,h5_name):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    if FLAGS.is_train:
        # savepath = os.path.join(os.getcwd(), os.path.join('checkpoint',data_dir,'train.h5'))
        savepath = os.path.join('.', os.path.join(data_dir, h5_name))
        if not os.path.exists(os.path.join('.', os.path.join( data_dir))):
            os.makedirs(os.path.join('.', os.path.join(data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data) # 这里的创建的h5文件的数据对象只有一个了 上一个FusionGan的网络有两个数据对象 data和label

def prepare_data(dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
        data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
        data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
        data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    print("data length: ", len(data))
    return data

#制作h5的主函数
def input_setup(config, data_dir,h5_name):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path 获取所有的图片路径
    if config.is_train:
        data = prepare_data(dataset=data_dir)

    sub_input_sequence = [] # 切割出来的每一张图片是其中的每一项数据

    if config.is_train:
        for i in range(len(data)): # 循环每一张图片
            #input_ = (imread(data[i]) - 127.5) / 127.5    归一化[-1,1]
            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            for x in range(0, h - config.image_size + 1, config.stride): # stride：24 对图像按照24步进行切割 每张图片的大小是128
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                    # Make channel value
                    if data_dir == "Train":
                        # sub_input = cv2.resize(sub_input, (config.image_size / 4, config.image_size / 4),
                        #                        interpolation=cv2.INTER_CUBIC)
                        # sub_input = sub_input.reshape([config.image_size / 4, config.image_size / 4, 1])
                        print('error')
                    else:
                        sub_input = sub_input.reshape([config.image_size, config.image_size, 1]) 

                    sub_input_sequence.append(sub_input)

    """
    len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
    (sub_input_sequence[0]).shape : (33, 33, 1)
    """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)  # [?, 33, 33, 1]
    # print(arrdata.shape)
    make_data(arrdata, data_dir,h5_name)

if __name__=="__main__":
    data_dir=""
    input_setip(FLAGS,)