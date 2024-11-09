import os
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader
from tools.accumulator import Accumulator
from wimans import WiMANS
from timesformer.models.vit import MyTimeSformer
from tools.logger import Logger


def accuracy(y1, y2, y1_hat, y2_hat):
    batch_size, num_users, num_classes = y2.shape

    y1, y1_hat = torch.argmax(y1, dim=2), torch.argmax(y1_hat, dim=2)
    y2, y2_hat = torch.argmax(y2, dim=2), torch.argmax(y2_hat, dim=2)

    acc1 = float((y1 == y1_hat).type(y1.dtype).sum() / (batch_size * num_users))
    acc2 = float((y2 == y2_hat).type(y1.dtype).sum() / (batch_size * num_users))

    return acc1, acc2


def evaluate(net, data_iter, loss_func):
    # y1训练损失之和, y2训练损失之和, 训练损失之和, y1正确预测的样本数, y2正确预测的样本数, 样本数
    metric = Accumulator(6)
    device = next(iter(net.parameters())).device
    net.eval()
    with torch.no_grad():
        for x, y1, y2 in data_iter:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            y1_hat, y2_hat = net(x)
            loss1, loss2 = loss_func(y1_hat, y1), loss_func(y2_hat, y2)
            loss = loss1 + loss2
            with torch.no_grad():
                eval_acc1, eval_acc2 = accuracy(y1, y2, y1_hat, y2_hat)
            metric.add(loss1.item() * x.shape[0], loss2.item() * x.shape[0], loss.item() * x.shape[0],
                       eval_acc1 * x.shape[0], eval_acc2 * x.shape[0], x.shape[0])

    return (metric[0] / metric[5], metric[1] / metric[5],
            metric[2] / metric[5], metric[3] / metric[5],
            metric[4] / metric[5])


def train(net, train_iter, eval_iter, learning_rate, num_epochs, patience, devices, checkpoint_save_dir_path, logger):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    # 在多个GPU上并行训练模型
    # net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    net = net.to(devices[0])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()

    best_state_dict = net.state_dict()
    min_eval_loss = float('inf')
    min_eval_loss_epoch = 0
    current_patience = 0

    for epoch in range(num_epochs):
        # y1训练损失之和, y2训练损失之和, 训练损失之和, y1正确预测的样本数, y2正确预测的样本数, 样本数
        metric = Accumulator(6)
        net.train()
        for i, (x, y1, y2) in enumerate(train_iter):
            optimizer.zero_grad()
            x, y1, y2 = x.to(devices[0]), y1.to(devices[0]), y2.to(devices[0])
            y1_hat, y2_hat = net(x)
            loss1, loss2 = loss_func(y1_hat, y1), loss_func(y2_hat, y2)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_acc1, train_acc2 = accuracy(y1, y2, y1_hat, y2_hat)
                metric.add(loss1.item() * x.shape[0], loss2.item() * x.shape[0], loss.item() * x.shape[0],
                           train_acc1 * x.shape[0], train_acc2 * x.shape[0], x.shape[0])
        train_loss1, train_loss2, train_loss, train_acc1, train_acc2 = (metric[0] / metric[5], metric[1] / metric[5],
                                                                        metric[2] / metric[5], metric[3] / metric[5],
                                                                        metric[4] / metric[5])
        eval_loss1, eval_loss2, eval_loss, eval_acc1, eval_acc2 = evaluate(net, eval_iter, loss_func)
        logger.record([
            f'Epoch: {epoch}, current patience: {current_patience + 1}',
            f'train loss1: {train_loss1:.3f}, train loss2: {train_loss2:.3f}, train loss: {train_loss:.3f}',
            f'train acc1: {train_acc1:.3f}, train acc2: {train_acc2:.3f}',
            f'eval loss1: {eval_loss1:.3f}, eval loss2: {eval_loss2:.3f}, eval loss: {eval_loss:.3f}',
            f'eval acc1: {eval_acc1:.3f}, eval acc2: {train_acc2:.3f}'
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


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_save_path = os.path.join('./', 'outputs', timestamp)
    logger = Logger(save_path=output_save_path)

    # devices = [torch.device('cuda:0')]
    devices = [torch.device("cpu")]

    npy_dir = r'E:\WorkSpace\WiMANS\dataset\wifi_csi\amp'  # .npy 文件所在的目录
    csv_file = r'E:\WorkSpace\WiMANS\dataset\annotation.csv'  # CSV 文件路径
    train_iter = DataLoader(WiMANS(npy_dir, csv_file), batch_size=2, shuffle=True)
    eval_iter = DataLoader(WiMANS(npy_dir, csv_file), batch_size=1, shuffle=True)

    net = MyTimeSformer()

    train(net, train_iter, eval_iter, 0.0001, 200, 10, devices, output_save_path, logger)