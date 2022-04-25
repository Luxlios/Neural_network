# -*- coding = utf-8 -*-
# @Time : 2022/4/23 10:06
# @Author : Luxlios
# @File : LSTM.py
# @Software : PyCharm

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super(lstm, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
    def forward(self, x):
        x, (hn, cn) = self.rnn(x)
        x = self.fc(x)
        return x

# def price2areturn(price):
#     '''
#     :param price: closing price
#     :return: arithmetic return
#     '''
#     areturn = [[1]]  # first arithmetic return = 1
#     for i in range(len(price)-1):
#         areturn.append([price[i+1][0]/price[i][0]])
#     return areturn


def dataset(data, seq_len):
    '''
    :param data: list
    :param seq_len: sequence length
    :return: sequence, label  (np.array)
    '''
    # normalization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mean) / std
    data = data.tolist()

    sequence, label = [], []
    for i in range(len(data)-seq_len):
        sequence.append(data[i:i+seq_len])
        label.append(data[i+seq_len])
    return torch.Tensor(sequence).float().permute([1, 0, 2]), torch.Tensor(label).float(), mean, std
    # sequence: [seq_len, batch_size, input_size]
    # label: [batch_size, output_size]

if __name__ == '__main__':
    '''
    network = lstm(input_size=1, hidden_size=2, output_size=1, num_layers=1)
    # [seq_len, batch_size, input_size]
    x = torch.FloatTensor(7, 20, 1)
    x = network(x)
    print(x.size())
    '''

    # dataset
    data = pd.read_csv('./BCHAIN-MKPRU.csv')
    # data = {'Data', 'Value'}
    # get arithmetic return
    # equivalent to data standardization, which is conducive to network convergence
    del data['Date']
    price = data.values.tolist()
    # a_return = price2areturn(price)  # data: arithmetic return

    seq_len = 14
    # sequence: [seq_len, batch_size(total), input_size]
    # label: [batch_size(total), output_size]
    sequence, label, mean, std = dataset(data=price, seq_len=seq_len)

    # network
    device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    network = lstm(input_size=1, hidden_size=20, output_size=1, num_layers=2)
    network = network.to(device)

    # optimizer & loss function
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    # train
    epoches = 50
    minibatch = 20   # batch_size = 20
    loss_record = []
    for epoch in range(epoches):
        network.train()
        loss_total = 0
        loss_num = 0
        for iteration in range(math.ceil(sequence.size(1)//minibatch)):
            inputs = sequence[:, iteration * minibatch:iteration * minibatch + minibatch, :]
            outputs = label[iteration * minibatch:iteration * minibatch + minibatch, :]
            inputs, outputs = inputs.to(device), outputs.to(device)
            # inputs: [seq_len, batch_size, input_size]
            # outputs: [batch_size, output_size]

            optimizer.zero_grad()
            predictions = network(inputs)[-1]  # last output

            loss = loss_function(predictions, outputs)
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            loss_num += inputs.size(1)
        loss_record.append(loss_total/loss_num)
        print('[epoch %d]loss: %.3f'%(epoch + 1, loss_total/loss_num))

    # figure
    # loss
    plt.figure()
    plt.plot(range(1, epoches + 1), loss_record, c='C0')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss(epoch=%d)'%epoches)

    # prediction
    network.eval()
    with torch.no_grad():
        label_predction = network(sequence.to(device))[-1]
    plt.figure()
    # de_normalization: label * std / mean
    plt.plot(range(sequence.size(1)), label.numpy() * std + mean, c='C0')
    plt.plot(range(sequence.size(1)), label_predction.cpu().detach().numpy() * std + mean, c='C1')
    plt.legend(['ground truth', 'prediction'])
    plt.xlabel('data')
    plt.ylabel('arithmetic return')
    plt.title('arithmetic return(epoch=%d)'%epoches)
    plt.show()
