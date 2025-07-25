# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # x和y是两个特征图，它们的形状都是(batch_size, channels, height, width)
# x = torch.randn(1, 256, 32, 32)
# y = torch.randn(1, 256, 32, 32)
#
# # 定义两个1x1卷积层
# conv1x1_x = nn.Conv2d(256, 256, 1)
# conv1x1_y = nn.Conv2d(256, 256, 1)
#
# # 通过卷积层得到新的特征图
# x_conv = conv1x1_x(x)
# y_conv = conv1x1_y(y)
#
# # 进行全局平均池化
# x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(1, -1)
# y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(1, -1)
#
# # 定义两个全连接层
# fc_x = nn.Linear(256, 256)
# fc_y = nn.Linear(256, 256)
#
# # 通过全连接层得到新的向量
# x_fc = fc_x(x_avgpool)
# y_fc = fc_y(y_avgpool)
#
# # 使用Sigmoid激活函数
# x_sigmoid = torch.sigmoid(x_fc).unsqueeze(-1).unsqueeze(-1)
# y_sigmoid = torch.sigmoid(y_fc).unsqueeze(-1).unsqueeze(-1)
#
# # 使用得到的向量和原始特征图进行注意力操作
# x_attention = x * y_sigmoid
# y_attention = y * x_sigmoid
#
# # 拼接两个特征图
# concatenated = torch.cat((x_attention, y_attention), dim=1)  # shape: (1, 512, 32, 32)
#
# # 使用一个1x1卷积来降维
# conv1x1_concat = nn.Conv2d(512, 256, 1)
# output = conv1x1_concat(concatenated)
#
# print(output.shape)
#
#


# 在这个类中，我们首先在构造函数__init__中定义了所有的需要的层。然后，在forward函数中，我们定义了如何通过这些层进行前向传播。在这个例子中，我们的模型接收两个输入x和y，并返回一个输出。

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossAttention, self).__init__()

        self.conv1x1_x = nn.Conv2d(in_channels, out_channels, 1)
        self.conv1x1_y = nn.Conv2d(in_channels, out_channels, 1)

        self.fc_x = nn.Linear(out_channels, out_channels)
        self.fc_y = nn.Linear(out_channels, out_channels)

        self.conv1x1_concat = nn.Conv2d(2*out_channels, out_channels, 1)

    def forward(self, x, y):
        x_conv = self.conv1x1_x(x)
        y_conv = self.conv1x1_y(y)

        x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(x.size(0), -1)
        y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(y.size(0), -1)

        x_fc = self.fc_x(x_avgpool)
        y_fc = self.fc_y(y_avgpool)

        x_sigmoid = torch.sigmoid(x_fc).view(x.size(0), -1, 1, 1)
        y_sigmoid = torch.sigmoid(y_fc).view(y.size(0), -1, 1, 1)

        x_attention = x * y_sigmoid
        y_attention = y * x_sigmoid

        concatenated = torch.cat((x_attention, y_attention), dim=1)  
        output = self.conv1x1_concat(concatenated)

        return output

# Usage
model = CrossAttention(256, 256)
x = torch.randn(1, 256, 32, 32)
y = torch.randn(1, 256, 32, 32)
output = model(x, y)
print(output.shape)





# 在这个示例中，我们首先导入了我们的CrossAttention类。然后，我们创建了一个CrossAttention的实例，叫做cross_attention。然后我们创建了两个输入x和y，然后把它们传入到我们的cross_attention实例中，得到一个输出。最后我们打印了输出。

# 注意这个例子假设你已经把CrossAttention类定义在一个.py文件中，叫做my_module.py。如果你的CrossAttention类是在其他地方定义的，你可能需要修改导入语句来正确地导入你的CrossAttention类。




#
# import torch
#
# # 首先，我们需要在文件顶部导入我们的CrossAttention类。这需要你将CrossAttention类定义在一个.py文件中，并且这个文件应在你的PYTHONPATH里面。
# # 假设我们的CrossAttention类被定义在一个叫做my_module的.py文件中
# from my_module import CrossAttention
#
# # 然后，我们可以实例化这个类。注意你需要提供对应的参数，这个参数应该和你的CrossAttention的初始化函数__init__需要的参数一致。
# cross_attention = CrossAttention(256, 256)
#
# # 现在我们有了一个CrossAttention的实例，我们可以使用这个实例来进行前向传播。
# # 例如，我们有两个输入x和y：
# x = torch.randn(1, 256, 32, 32)
# y = torch.randn(1, 256, 32, 32)
#
# # 我们可以调用我们的cross_attention实例，并把x和y作为输入进行前向传播
# output = cross_attention(x, y)
#
# # 输出应该是我们的CrossAttention类中定义的输出。在这个例子中，它应该是一个Tensor
# print(output)


import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()

        self.conv1x1_x = nn.Conv2d(0, 0, 1)  # 初始化卷积层，后续会更新权重
        self.conv1x1_y = nn.Conv2d(0, 0, 1)  # 初始化卷积层，后续会更新权重

        self.fc_x = nn.Linear(0, 0)  # 初始化全连接层，后续会更新权重
        self.fc_y = nn.Linear(0, 0)  # 初始化全连接层，后续会更新权重

        self.conv1x1_concat = nn.Conv2d(0, 0, 1)  # 初始化卷积层，后续会更新权重

    def forward(self, x, y):
        channels = x.size(1)  # 获取输入的通道数

        # 动态创建网络层，根据输入通道数更新权重
        if not isinstance(self.conv1x1_x, nn.Conv2d):
            self.conv1x1_x = nn.Conv2d(channels, channels, 1)
            self.conv1x1_y = nn.Conv2d(channels, channels, 1)
            self.fc_x = nn.Linear(channels, channels)
            self.fc_y = nn.Linear(channels, channels)
            self.conv1x1_concat = nn.Conv2d(2*channels, channels, 1)

        x_conv = self.conv1x1_x(x)
        y_conv = self.conv1x1_y(y)

        x_avgpool = F.adaptive_avg_pool2d(x_conv, (1, 1)).view(x.size(0), -1)
        y_avgpool = F.adaptive_avg_pool2d(y_conv, (1, 1)).view(y.size(0), -1)

        x_fc = self.fc_x(x_avgpool)
        y_fc = self.fc_y(y_avgpool)

        x_sigmoid = torch.sigmoid(x_fc).view(x.size(0), -1, 1, 1)
        y_sigmoid = torch.sigmoid(y_fc).view(y.size(0), -1, 1, 1)

        x_attention = x * y_sigmoid
        y_attention = y * x_sigmoid

        concatenated = torch.cat((x_attention, y_attention), dim=1)
        output = self.conv1x1_concat(concatenated)

        return output

# Usage
model = CrossAttention()
x = torch.randn(1, 256, 32, 32)
y = torch.randn(1, 256, 32, 32)
output = model(x, y)
print(output.shape)