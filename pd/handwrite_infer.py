# 加载飞桨和相关类库
import paddle
import net_base as nb
from PIL import Image
import numpy as np
import os


# 读取一张本地的样例图片，转变成模型输入的格式


def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图,黑色是0白色是255
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    # im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # t1 = np.array(im)
    # 颜色反转
    elements = [0 if elem > 100 else 255 for row in np.array(im) for elem in row]
    im = np.array(elements).reshape(1, 1, 28, 28).astype(np.float32)
    # 图像归一化
    im = 1.0 - im / 255.
    return im


def infer():
    # 定义预测过程
    # model = nb.LeNet(num_classes=10)
    model = nb.MNIST()
    params_file_path = 'mnist_200_200.pdparams'
    # test_212_9
    # test_213_3
    # test_328_7
    # test_706_9
    # test_1107_9
    # test_1151_6
    img_path = 'C:/360Downloads\迅雷下载/test/test_213_3.jpg'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    # 灌入数据
    model.eval()
    tensor_img = load_image(img_path)
    # 模型反馈10个分类标签的对应概率
    results = model(paddle.to_tensor(tensor_img))


def get_ret(results):
    # 取概率最大的标签作为预测输出
    nx = results.numpy()
    # print(nx)
    # [[  4.65034    -6.0972753   8.65909    -5.2514696  -8.976308  -15.372444 -5.774998  -12.30933    13.431808
    # 6.3811927]] np.argsort 将下标按大小排序， [[5 7 4 1 6 3 0 9 2 8]]; 13.431808 为最大，是第8个，故用lab[0][-1]得到概率最大的那个即下标为8，对应的就是8
    # [0,1,2,3,4,5,6,7,8,9]
    lab = np.argsort(nx)
    # print(lab)
    # print("本次预测的数字是: ", lab[0][-1])
    return lab[0][-1]


def read_photo():
    # alexNet mnist_100_20 accuracy: 9649 / 10000
    # model = nb.AlexNet(10)
    # resnet mnist_120_2 accuracy: 9303 / 10000
    # resnet mnist_120_20  accuracy: 9651 / 10000
    # resnet mnist_120_30  accuracy: 9771 / 10000
    model = nb.ResNet(layers=50, class_dim=10)
    params_file_path = 'mnist_120_30.pdparams'
    img_path = 'C:/360Downloads\迅雷下载/test'
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)
    # 灌入数据
    model.eval()

    # 模型反馈10个分类标签的对应概率
    total = 0
    correct = 0
    for file in os.listdir(img_path):
        tensor_img = load_image(img_path + '/' + file)
        # nb.print_img(tensor_img[0][0])
        images = paddle.to_tensor(tensor_img)
        results = model(images)
        cur = int(file.split('.')[0].split('_')[-1])
        ret = get_ret(results)
        print('file: %s : %d : %d' % (file, cur, ret))
        total = total + 1
        if cur == ret:
            correct = correct + 1
    print('accuracy: %d / %d' % (correct, total))


# infer()
read_photo()
