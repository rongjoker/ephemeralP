import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import d2l_torch as d2l
import cnn_base as base


# 0.9375 batch_size 256 max_epochs 20
def vit_train():
    img_size, patch_size = 96, 16
    num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
    emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
    model = base.ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
                     num_blks, emb_dropout, blk_dropout, lr)
    trainer = d2l.Trainer(max_epochs=20, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=256, resize=(img_size, img_size))
    trainer.fit(model, data)
    torch.save(model.state_dict(), 'model/fashion_mnist_vit.pth')


def resnet_train():
    # 10 epoch acc: tensor(0.9375, device='cuda:0')
    base.ResNet18().layer_summary((1, 1, 96, 96))

    model = base.ResNet18(lr=0.01)
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    trainer.fit(model, data)
    torch.save(model.state_dict(), 'model/fashion_mnist_resnet.pth')


# 定义一个函数来加载并预处理图像
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # image = Image.open(image_path)
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


# resnet_train()
# resnet_infer()
# vit_train()
vit_infer()

