import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import d2l_torch as d2l
import cnn_base as base
import numpy as np
import os


def vit_train(epoch=10, batch_size=128):
    img_size, patch_size = 96, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    model = base.ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                     num_blks, emb_dropout, blk_dropout, lr)
    trainer = d2l.Trainer(max_epochs=epoch, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(img_size, img_size))
    trainer.fit(model, data)
    torch.save(model.state_dict(), 'model/fashion_mnist_vit.pth')


def resnet_train(epoch=10, batch_size=128):
    # 10 epoch acc: tensor(0.9375, device='cuda:0')
    base.ResNet18().layer_summary((1, 1, 96, 96))

    model = base.ResNet18(lr=0.01)
    trainer = d2l.Trainer(max_epochs=epoch, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(96, 96))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)
    torch.save(model.state_dict(), 'model/fashion_mnist_resnet.pth')


# 定义一个函数来加载并预处理图像
# todo 对于图片将所有的灰度(0-255)全部映射到(0.01-0.99)上
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    image = Image.open(image_path).convert('L')  # 将图像转换为灰阶
    image = transform(image).unsqueeze(0)
    return image


def try_gpu():
    return torch.device('cuda:0')


def get_fashion_mnist_labels(labels):  # @save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def predict_ch3(net, test_iter, n=6):  # @save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    # todo
    X = X.to(try_gpu())
    first = X[0][0]
    print(first.argmax(axis=0))
    # 图像没有归一化，打印出来最大是这样的 66, 66, 66, 66, 67, 67, 67, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70
    print(X.shape)
    y = y.to(try_gpu())
    possibility = net(X)
    infer_ret = possibility.argmax(axis=1)
    preds = d2l.get_fashion_mnist_labels(infer_ret)
    trues = d2l.get_fashion_mnist_labels(y)
    # print(preds)
    # print(trues)
    index = 0
    for i, label in enumerate(preds):
        if label == trues[i]:
            index = index + 1
    print("%d/%d" % (index, len(trues)))
    # preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]

    # d2l.show_images(
    #     X.cpu()[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])


def resnet_infer():
    # 加载预训练的ResNet模型
    model = base.ResNet18(lr=0.01)
    model.load_state_dict(torch.load('model/fashion_mnist_resnet.pth'))
    model = model.to(try_gpu())
    # model = models.resnet50(pretrained=True)
    model.eval()  # 设置模型为推理模式

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=(96, 96))
    predict_ch3(model, test_iter)

    # 读取要分类的图像
    image_path = 'test/pant1.png'  # 替换成你的图像文件路径
    input_image = preprocess_image(image_path)
    print(input_image.shape)
    print(type(input_image))
    # 进行推理
    with torch.no_grad():
        output = model(input_image.to(try_gpu()))

    # 获取类别标签
    _, predicted_class = output.max(1)
    print(get_fashion_mnist_labels([predicted_class.item()]))


def vit_infer():
    img_size, patch_size = 96, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    model = base.ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                     num_blks, emb_dropout, blk_dropout, lr)
    model.load_state_dict(torch.load('model/fashion_mnist_vit.pth'))
    model = model.to(try_gpu())
    model.eval()  # 设置模型为推理模式

    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=(96, 96))
    predict_ch3(model, test_iter)

    # 读取要分类的图像
    image_path = 'test/pant1.png'  # 替换成你的图像文件路径
    input_image = preprocess_image(image_path)
    print(input_image.shape)
    print(type(input_image))
    # 进行推理
    with torch.no_grad():
        output = model(input_image.to(try_gpu()))

    # 获取类别标签
    _, predicted_class = output.max(1)
    print(get_fashion_mnist_labels([predicted_class.item()]))


def renet_infer_batch():
    # 加载预训练的ResNet模型
    model = base.ResNet18(lr=0.01)
    model.load_state_dict(torch.load('model/fashion_mnist_resnet.pth'))
    model = model.to(try_gpu())
    # model = models.resnet50(pretrained=True)
    model.eval()  # 设置模型为推理模式
    input_image_list = []
    for file in os.listdir('test'):
        print(file)
        out_ = preprocess_image(  'test/' + file)
        input_image_list.append(out_[0])
    ndarray_data = np.asarray(input_image_list)
    input_image = torch.from_numpy(ndarray_data)

    # 读取要分类的图像
    # image_path = 'test/pant1.png'  # 替换成你的图像文件路径
    # input_image = preprocess_image(image_path)
    # print(input_image.shape)
    # input_image = [input_image[0] for _ in (0, 10)]
    # ndarray_data = np.asarray(input_image)
    # input_image = torch.from_numpy(ndarray_data)
    # print(input_image.shape)
    # print(type(input_image))
    # todo
    # 单个是torch.Size([1, 1, 96, 96])
    # need torch.Size([256, 1, 96, 96])
    # 进行推理
    with torch.no_grad():
        output = model(input_image.to(try_gpu()))

    # 获取类别标签
    infer_ret = output.argmax(axis=1)
    print(d2l.get_fashion_mnist_labels(infer_ret))
    # _, predicted_class = output.max(1)
    # print(get_fashion_mnist_labels([predicted_class.item()]))


def get_img(path):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert('L')
    im = np.array(img)
    # ndarray_data = np.asarray(input_image)
    input_image = torch.from_numpy(im)
    print(input_image.shape)
    print(im.argmax(axis=0))
    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = transform(img)
    print(img[0].argmax(axis=0))
    print(img.shape)


# max_epochs=50 batch_size=256 0.9375
# max_epochs=50 batch_size=128
# resnet_train(epoch=50, batch_size=128)
# resnet_infer()
renet_infer_batch()
# 0.9375 batch_size 256 max_epochs 20
# vit_train(epoch=50, batch_size=256)
# vit_infer()
# get_img('test/pant1.png')
