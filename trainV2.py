import os
import datetime
import torch
from torch import nn
from tools.accumulator import Accumulator
from wimans import WiMANS, get_dataloaders
from timesformer.models.vit import MyTimeSformer, MyTimeSformerV2
from tools.logger import Logger
from .models.vit import VisionTransformer


def accuracy(y1, y2, y3, y1_hat, y2_hat, y3_hat):
    y1_hat = (y1_hat > 0.5).float()  # 对于身份识别，0.5为阈值
    identity_accuracy = (y1_hat == y1).float().mean().item()

    y2_hat = (y2_hat > 0.5).float()  # 对于位置识别，0.5为阈值
    location_accuracy = (y2_hat == y2).float().mean().item()

    y3_hat = (y3_hat > 0.5).float()  # 对于活动识别，0.5为阈值
    activity_accuracy = (y3_hat == y3).float().mean().item()

    return identity_accuracy, location_accuracy, activity_accuracy


def evaluate(net, data_iter, loss_func=nn.BCELoss()):
    # 训练损失之和, y1训练损失之和, y2训练损失之和, y3训练损失之和, 样本数, y1正确预测的样本数, y2正确预测的样本数, y3正确预测的样本数
    metric = Accumulator(8)
    device = next(iter(net.parameters())).device
    net.eval()

    with torch.no_grad():
        for i, (x, y1, y2, y3) in enumerate(data_iter):
            batch_size, num_users = y1.shape
            x, y1, y2, y3 = x.to(device), y1.to(device), y2.to(device), y3.to(device)
            y1_hat, y2_hat, y3_hat = net(x)

            loss1 = loss_func(y1_hat, y1)
            loss2 = loss_func(y2_hat, y2)
            loss3 = loss_func(y3_hat, y3)

            loss = loss1 + loss2 + loss3

            identity_acc, location_acc, activity_acc = accuracy(y1, y2, y3, y1_hat, y2_hat, y3_hat)
            metric.add(loss.item() * batch_size,
                       loss1.item() * batch_size, loss2.item() * batch_size, loss3.item() * batch_size,
                       batch_size, identity_acc * batch_size, location_acc * batch_size, activity_acc * batch_size)

    return (metric[0] / metric[4],
            metric[1] / metric[4], metric[2] / metric[4], metric[3] / metric[4],
            metric[5] / metric[4], metric[6] / metric[4], metric[7] / metric[4])


def train(net, train_iter, eval_iter, learning_rate, num_epochs, patience, devices, checkpoint_save_dir_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.BCELoss()

    best_state_dict = net.state_dict()
    min_eval_loss = float('inf')
    min_eval_loss_epoch = 0
    current_patience = 0

    for epoch in range(num_epochs):
        # 训练损失之和, y1训练损失之和, y2训练损失之和, y3训练损失之和, 样本数, y1正确预测的样本数, y2正确预测的样本数, y3正确预测的样本数
        metric = Accumulator(8)
        net.train()
        for i, (x, y1, y2, y3) in enumerate(train_iter):
            batch_size, num_users = y1.shape
            optimizer.zero_grad()
            x, y1, y2, y3 = x.to(devices[0]), y1.to(devices[0]), y2.to(devices[0]), y3.to(devices[0])
            y1_hat, y2_hat, y3_hat = net(x)

            loss1 = loss_func(y1_hat, y1)
            loss2 = loss_func(y2_hat, y2)
            loss3 = loss_func(y3_hat, y3)

            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                identity_acc, location_acc, activity_acc = accuracy(y1, y2, y3, y1_hat, y2_hat, y3_hat)
                metric.add(loss.item() * batch_size, loss1.item() * batch_size, loss2.item() * batch_size,
                           loss3.item() * batch_size,
                           batch_size, identity_acc * batch_size, location_acc * batch_size, activity_acc * batch_size)

                if i % 20 == 0:
                    train_loss, train_loss1, train_loss2, train_loss3 = (metric[0] / metric[4],
                                                                         metric[1] / metric[4], metric[2] / metric[4],
                                                                         metric[3] / metric[4])
                    print(
                        f'Epoch: {epoch}, iter: {i}, train loss: {train_loss:.3f}, train identity loss: {train_loss1:.3f}, train location loss2: {train_loss2:.3f}, train activity loss: {train_loss3:.3f}')

        train_loss, train_loss1, train_loss2, train_loss3, train_acc1, train_acc2, train_acc3 = (metric[0] / metric[4],
                                                                                                 metric[1] / metric[4],
                                                                                                 metric[2] / metric[4],
                                                                                                 metric[3] / metric[4],
                                                                                                 metric[5] / metric[4],
                                                                                                 metric[6] / metric[4],
                                                                                                 metric[7] / metric[4])

        eval_loss, eval_loss1, eval_loss2, eval_loss3, eval_acc1, eval_acc2, eval_acc3 = evaluate(net, eval_iter)

        logger.record([
            f'Epoch: {epoch}, current patience: {current_patience + 1}',
            f'train loss: {train_loss:.3f}, train identity loss: {train_loss1:.3f}, train location loss: {train_loss2:.3f}, train activity loss: {train_loss3:.3f}',
            f'train identity acc: {train_acc1:.3f}, train location acc: {train_acc2:.3f}, train activity acc: {train_acc3:.3f}',
            f'eval loss: {eval_loss:.3f}, eval identity loss: {eval_loss1:.3f}, eval location loss: {eval_loss2:.3f}, eval activity loss: {eval_loss3:.3f}',
            f'eval identity acc: {eval_acc1:.3f}, eval location acc: {eval_acc2:.3f}, eval activity acc: {eval_acc3:.3f}',
        ])

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            checkpoint_save_path = os.path.join(checkpoint_save_dir_path, f"checkpoint-{epoch}.pth")
            torch.save(net.state_dict(), checkpoint_save_path)
        if eval_loss < min_eval_loss:
            best_state_dict = net.state_dict()
            min_eval_loss = eval_loss
            min_eval_loss_epoch = epoch
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                logger.record([f'Early stopping after {epoch + 1} epochs'])
                break

    torch.save(best_state_dict, os.path.join(checkpoint_save_dir_path, "best_state_dict.pth"))
    logger.record([f"The best testing loss occurred in the {min_eval_loss_epoch} epoch"])

    return os.path.join(checkpoint_save_dir_path, "best_state_dict.pth")


def test(net, pth_path, test_iter, device, logger):
    net = torch.nn.DataParallel(net)
    net.load_state_dict(torch.load(pth_path))
    net.to(device)
    net.eval()

    test_loss, test_loss1, test_loss2, test_loss3, test_acc1, test_acc2, test_acc3 = evaluate(net, test_iter)
    logger.record([
        f'test loss: {test_loss:.3f}, test identity loss: {test_loss1:.3f}, test location loss: {test_loss2:.3f}, test activity loss: {test_loss3:.3f}',
        f'test identity acc: {test_acc1:.3f}, test location acc: {test_acc2:.3f}, test activity acc: {test_acc3:.3f}'
    ])


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'outputs', timestamp)
    logger = Logger(save_path=output_save_path)

    devices = [torch.device('cuda:0'), torch.device('cuda:1'), torch.device('cuda:2'), torch.device('cuda:3')]
    # devices = [torch.device("cpu")]

    npy_dir = '/home/dataset/XLBWorkSpace/wimans/wifi_csi/amp/'  # .npy 文件所在的目录
    csv_file = '/home/dataset/XLBWorkSpace/wimans/annotation.csv'  # CSV 文件路径
    dataset = WiMANS(npy_dir, csv_file)
    train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=16)

    # net = MyTimeSformer()
    # net = MyTimeSformerV2(img_size=96, num_classes=10, num_frames=3000, attention_type='divided_space_time')
    net = VisionTransformer()

    pth_path = train(net, train_loader, val_loader, 0.0001, 300, 20, devices, output_save_path, logger)

    test(net, pth_path, test_loader, devices[0], logger)
