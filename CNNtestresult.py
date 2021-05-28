import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
padding = 0
ndf = 18
alpha = 0.01

XTrain = np.load('XTrain.npy', allow_pickle=True)
YTrain = np.load('YTrain.npy', allow_pickle=True)
elapsed = time.time() - start
print("data load time used:", elapsed)

XTrain = torch.from_numpy(np.expand_dims(XTrain, 1)).float()
YTrain = torch.from_numpy(np.expand_dims(YTrain, 1)).long()

row_rand = np.arange(XTrain.shape[0])
np.random.shuffle(row_rand)

validation_size = (XTrain.shape[0] // 10) * 9
XTrain_all = XTrain[row_rand[0:validation_size]]
YTrain_all = YTrain[row_rand[0:validation_size]]

XTest = XTrain[row_rand[validation_size:]]
YTest = YTrain[row_rand[validation_size:]]

batch_size = 1


torch_testset = Data.TensorDataset(XTest, YTest)
loader2 = Data.DataLoader(
    dataset=torch_testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)


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

        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.BN1 = nn.BatchNorm2d(ndf * 2)

        self.conv5 = nn.Conv2d(ndf * 2, ndf, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.fc1 = nn.Linear(ndf, ndf // 2)
        self.fc2 = nn.Linear(ndf // 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.BN0(x))

        x = self.conv2(x)
        x = F.relu(self.BN1(x))

        x = self.conv5(x)
        spp = self.spatial_pyramid_pool(x, x.size()[0], [int(x.size(2)), int(x.size(3))], self.output_num)

        fc1 = self.fc1(spp)
        fc2 = self.fc2(fc1)

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


VGGSPP = torch.load('/home/yinyibo/PycharmProjects/pytorch/ECG/' +
                    'code/11.0/ICCSN/paper/CNN_validation_best/9_epoch120_0.94708_0.9480_0.9416.pth')

print(VGGSPP)

summary(VGGSPP, (1, XTrain.shape[2], XTrain.shape[3]))  # 65 33
input_t = torch.randn(1, 1, XTrain.shape[2], XTrain.shape[3]).cuda()
flops, params = profile(VGGSPP, inputs=(input_t,))
print(flops, params)


for step2, (batch_x2, batch_y2) in enumerate(loader2):
    print('valid:step %d' % step2)
    x2 = batch_x2.cuda()
    if step2 == 0:
        pred_test = torch.max(VGGSPP(x2), 1)
        pred_test = pred_test[1]
    else:
        pred_test = torch.cat((pred_test, torch.max(VGGSPP(x2), 1)[1]), dim=0)

pred_test = pred_test.cpu().numpy()
np.save('YT2_1.npy', pred_test)
temp = np.expand_dims(pred_test, 1) == YTest.numpy()
accuracy = np.sum(temp) / np.size(temp)


print('accuracy_all:%f' % accuracy)

abnormal_right = 0
normal_right = 0
for i in range(len(pred_test)):
    if YTest[i] == 1:
        abnormal_right += pred_test[i] == 1
    else:
        normal_right += pred_test[i] == 0

accuracy_normal = (normal_right / (len(YTest) - sum(YTest)))
accuracy_abnormal = (abnormal_right / (sum(YTest)))
print('accuracy_normal:%f' % (accuracy_normal))
print('accuracy_abnormal:%f' % (accuracy_abnormal))

