import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torch.utils.data as Data
import time
import os
import datetime
from focalloss import FocalLoss
from torchsummary import summary
from thop import profile
from matplotlib import pyplot as plt

start = time.time()
np.random.seed(1337)

kernel_size = 4
stride = 2
padding = 2
ndf = 18
alpha = 0.01
XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
print("data load time used:", elapsed)

np.random.seed(1337)
XTrain = torch.from_numpy(np.expand_dims(XTrain, 1)).float()
YTrain = torch.from_numpy(np.expand_dims(YTrain, 1)).long()

row_rand = np.arange(XTrain.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain.shape[0]//10)*9
XTrain_all = XTrain[row_rand[0:validation_size]]
YTrain_all = YTrain[row_rand[0:validation_size]]

XTest = XTrain[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]

valid_num = len(XTrain_all)//10

for valid_index in range(10):
    XValid = XTrain_all[valid_num * valid_index:valid_num * (valid_index + 1)]
    YValid = YTrain_all[valid_num * valid_index:valid_num * (valid_index + 1)]
    temp = list(range(len(XTrain_all)))
    del temp[valid_num * valid_index:valid_num * (valid_index + 1)]
    XTrain_actual = XTrain_all[temp]
    YTrain_actual = YTrain_all[temp]

    elapsed = time.time() - start
    print("data transform&choose and time used:", elapsed)
    start = time.time()

    batch_size = 256 * 2

    torch_dataset = Data.TensorDataset(XTrain_actual, YTrain_actual)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    YValid = YTest
    torch_testset = Data.TensorDataset(XTest, XTest)
    loader2 = Data.DataLoader(
        dataset=torch_testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    elapsed = time.time() - start
    print("data to dataloader time used:", elapsed)

    class SPP_NET(nn.Module):
        '''
        A CNN model which adds spp layer so that we can input multi-size tensor
        '''

        def __init__(self, opt, input_nc, ndf, gpu_ids=[]):
            super(SPP_NET, self).__init__()
            self.gpu_ids = gpu_ids
            self.output_num = [1]

            self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False)
            self.BN0 = nn.BatchNorm2d(ndf)

            self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False)
            self.BN1 = nn.BatchNorm2d(ndf*2)

            # self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=kernel_size, stride=stride,
            #                        padding=padding,  bias=False)
            # self.BN2 = nn.BatchNorm2d(ndf * 4)
            #
            # self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=kernel_size, stride=stride,
            #                        padding=padding,  bias=False)
            # self.BN3 = nn.BatchNorm2d(ndf * 8)

            self.conv5 = nn.Conv2d(ndf*2, ndf, kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False)

            self.fc1 = nn.Linear(ndf, ndf//2)
            self.fc2 = nn.Linear(ndf//2, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(self.BN0(x))

            x = self.conv2(x)
            x = F.relu(self.BN1(x))

            # x = self.conv3(x)
            # x = F.leaky_relu(self.BN2(x))
            #
            # x = self.conv4(x)
            # x = F.leaky_relu(self.BN3(x))
            x = self.conv5(x)
            spp = self.spatial_pyramid_pool(x, x.size()[0], [int(x.size(2)), int(x.size(3))], self.output_num)
            # print(spp.size())
            fc1 = self.fc1(spp)
            fc2 = self.fc2(fc1)
            # s = nn.Softmax()
            # output = s(fc2)
            output = fc2
            return output

        def spatial_pyramid_pool(self, previous_conv, num_sample, previous_conv_size, out_pool_size):
            '''
            previous_conv: a tensor vector of previous convolution layer
            num_sample: an int number of image in the batch
            previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
            out_pool_size: a int vector of expected output size of max pooling layer

            returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
            '''
            # print(previous_conv.size())
            for i in range(len(out_pool_size)):
                # print(previous_conv_size)
                h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
                w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
                h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) // 2
                w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) // 2
                maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
                x = maxpool(previous_conv)
                if (i == 0):
                    spp = x.view(num_sample, -1)
                    # print("spp size:",spp.size())
                else:
                    # print("size:",spp.size())
                    spp = torch.cat((spp, x.view(num_sample, -1)), 1)
            return spp

    VGGSPP = SPP_NET(1, 1, ndf=ndf, gpu_ids=[0, 1, 2, 3]).cuda()
    # VGGSPP = nn.DataParallel(VGGSPP)
    # VGGSPP = VGGSPP.cuda()

    print(VGGSPP)

    summary(VGGSPP, (1, XTrain.shape[2], XTrain.shape[3])) #65 33
    input_t = torch.randn(1, 1, XTrain.shape[2], XTrain.shape[3]).cuda()
    flops, params = profile(VGGSPP, inputs=(input_t, ))
    print(flops, params)

    optimizer = torch.optim.Adam(VGGSPP.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    accuracy = [0]
    accuracy_normal = [0]
    accuracy_abnormal = [0]
    max_accuracy = 0
    loss_all = [0]
    dt = datetime.datetime.now()
    # kernel_size = 3
    # stride = 2
    # padding = 1
    # ndf = 16
    model_dir = 'train_parallel_{:0>2d}_{:0>2d}_{:0>2d}_{:0>2d}_{:0>4f}_{:0>2d}'.format(kernel_size, stride, padding, ndf, alpha, valid_index)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    for epoch in range(201):
        start = time.time()
        print('epoch:%d' % epoch)
        for step, (batch_x, batch_y) in enumerate(loader):
            x = batch_x.cuda()
            y = batch_y.cuda().squeeze()
            print('step:%d' % step)
            out = VGGSPP(x)
            # loss = loss_func(out, y)
            loss = FocalLoss(gamma=2, alpha=[alpha, 1])(out, y)
            loss_all.append(loss.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for step2, (batch_x2, batch_y2) in enumerate(loader2):
            print('valid:step %d' % step2)
            x2 = batch_x2.cuda()
            if step2 == 0:
                pred_test = torch.max(VGGSPP(x2), 1)
                pred_test = pred_test[1]
            else:
                pred_test = torch.cat((pred_test, torch.max(VGGSPP(x2), 1)[1]), dim=0)

        pred_test = pred_test.cpu().numpy()
        temp = np.expand_dims(pred_test, 1) == YValid.numpy()
        temp = np.sum(temp)/np.size(temp)
        accuracy.append(temp)
        print('accuracy_all:%f' % accuracy[-1])

        abnormal_right = 0
        normal_right = 0
        for i in range(len(pred_test)):
            if YValid[i] == 1:
                abnormal_right += pred_test[i] == 1
            else:
                normal_right += pred_test[i] == 0

        accuracy_normal.append(normal_right/(len(YValid)-sum(YValid)))
        accuracy_abnormal.append(abnormal_right/(sum(YValid)))
        print('accuracy_normal:%f' % (accuracy_normal[-1]))
        print('accuracy_abnormal:%f' % (accuracy_abnormal[-1]))

        if max_accuracy < temp:
            torch.save(VGGSPP, './'+model_dir+'/epoch' + str(epoch) + '_'
                       + str(accuracy[-1])[0:7] + '_' +
                       str((normal_right/(len(YValid)-sum(YValid))).numpy())[1:7] + '_'
                       + str((abnormal_right/(sum(YValid))).numpy())[1:7] + '.pth')
            max_accuracy = temp
        if epoch % 20 == 0:
            torch.save(VGGSPP, './'+model_dir+'/epoch' + str(epoch) + '_'
                       + str(accuracy[-1])[0:7] + '_' +
                       str((normal_right/(len(YValid)-sum(YValid))).numpy())[1:7] + '_'
                       + str((abnormal_right/(sum(YValid))).numpy())[1:7] + '.pth')
        elapsed = time.time() - start
        print("epoch:", epoch, "time used:", elapsed)

    np.save('./'+model_dir+'/accuracy', accuracy)
    np.save('./'+model_dir+'/accuracy_normal', accuracy_normal)
    np.save('./'+model_dir+'/accuracy_abnormal', accuracy_abnormal)

# send email to me
# import send_email
# send_m = send_email.Sdem()
# send_email.Sdem.pro_over_send(send_m)

