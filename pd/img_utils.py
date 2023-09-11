# import handwrite_main as hm
import paddle
import net_base as nb
import numpy as np
import matplotlib.pyplot as plt


def evaluation(model, datasets):
    model.eval()

    acc_set = list()
    for batch_id, data in enumerate(datasets()):
        images, labels = data
        ts = np.array(images[0][0])
        # print_img(images[0][0])
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        pred = model(images)  # 获取预测值
        pn = pred.numpy()
        lab = np.argsort(pn)
        for i, la in enumerate(labels):
            print('%d ：%d' % (lab[i][-1], la.item()))
        acc = paddle.metric.accuracy(input=pred, label=labels)
        nx = acc.numpy()
        acc_set.extend(nx)

    # #计算多个batch的准确率
    acc_val_mean = np.array(acc_set).mean()
    print(' val acc: {}'.format(acc_val_mean))
    return acc_val_mean


val_dataset = nb.MnistDataset(mode='valid')
val_loader = paddle.io.DataLoader(val_dataset, batch_size=10, drop_last=True)
mod = nb.MNIST()
params_file_path = 'mnist_200_200.pdparams'
param_dict = paddle.load(params_file_path)
mod.load_dict(param_dict)

evaluation(model=mod, datasets=val_loader)
