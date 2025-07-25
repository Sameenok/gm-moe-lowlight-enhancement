import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.models.archs.SwinT import SwinT
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv_stride2(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,stride=2,
        padding=(kernel_size//2), bias=bias)
# 简单的门控机制GateWeight
class GateWeight(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # 将输入的x分成两份
        return x1 * x2


# Expert1---------专家1
class Expert1(nn.Module):
    def __init__(self, n_feats):
        super(Expert1, self).__init__()
        f = n_feats // 4  # 将输入特征数量 n_feats 除以 4，得到 f
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)  # 1×1卷积，用于特征提取
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)  # 1×1卷积，用于后续处理
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)  # 3×3卷积，带有填充
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)  # 步长为2的3×3卷积
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)  # 3×3卷积，带有填充
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)  # 另一个3×3卷积，带有填充
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)  # 1×1卷积，将特征映射回原始维度
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数，使用原地操作

    def forward(self, x):  # 定义前向传播方法，输入特性 x
        c1_ = self.conv1(x)  # 通过 1×1 卷积提取特征
        c1 = self.conv2(c1_)  # 经过步长为2的卷积层
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # 最大池化操作
        v_range = self.relu(self.conv_max(v_max))  # 经过 3×3 卷积和 ReLU 激活
        c3 = self.relu(self.conv3(v_range))  # 经过 3×3 卷积和 ReLU 激活
        c3 = self.conv3_(c3)  # 再次经过另一个 3×3 卷积
        # 使用双线性插值将特征图上采样到输入 x 的大小
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)  # 经过 1×1 卷积的处理
        c4 = self.conv4(c3 + cf)  # 将 c3 和 cf 逐元素相加
        m = self.sigmoid(c4)  # 应用 Sigmoid 激活函数

        return x * m  # 返回输入 x 与 m 的逐元素乘积

############################################################################################
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # 全局平均池化：将特征映射为单点
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通过减少和增加特征通道的卷积操作来获取通道权重
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y  # 返回输入x与通道注意力加权后的结果
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 使用卷积层来获取空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 将平均值和最大值拼接
        x = self.conv1(x)  # 通过卷积处理拼接后的结果
        y = self.sigmoid(x)  # 应用Sigmoid激活函数获取注意力权重
        return y * res  # 返回输入x与空间注意力加权后的结果

class Expert2(nn.Module):
    def __init__(self, n_feats):
        super(Expert2, self).__init__()
        # 定义一系列卷积层
        self.c1 = default_conv(n_feats, n_feats, 1)
        self.c2 = default_conv(n_feats, n_feats // 2, 3)
        self.c3 = default_conv(n_feats, n_feats // 2, 3)
        self.c4 = default_conv(n_feats*2, n_feats, 3)
        self.c5 = default_conv(n_feats // 2, n_feats // 2, 3)
        self.c6 = default_conv(n_feats*2, n_feats, 1)
        # 通道注意力层
        self.se = CALayer(channel=2*n_feats, reduction=16)
        # 空间注意力层
        self.sa = SpatialAttention()
        # 激活函数
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        res = x  # 保存原始输入
        # 通过一系列卷积层和激活函数处理输入
        y1 = self.act(self.c1(x))
        y2 = self.act(self.c2(y1))
        y3 = self.act(self.c3(y1))
        # 将不同层的输出进行拼接
        cat1 = torch.cat([y1, y2, y3], 1)
        y4 = self.act(self.c4(cat1))
        y5 = self.c5(y3)
        cat2 = torch.cat([y2, y5, y4], 1)
        # 通过通道注意力层和空间注意力层处理拼接后的输出
        ca_out = self.se(cat2)
        sa_out = self.sa(cat2)
        # 将通道注意力输出和空间注意力输出相加
        y6 = ca_out + sa_out
        # 经过最后一个卷积层
        y7 = self.c6(y6)
        # 将最终的输出与原始输入相加
        output = res + y7
        return output

# 通道注意力
class CA(nn.Module):
    def __init__(self, num_fea):
        super(CA, self).__init__()
        self.conv_du = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_fea, num_fea // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_fea // 8, num_fea, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, fea):
        return self.conv_du(fea)

#空间注意力模块
class SA(nn.Module):
    def __init__(self, n_feats, conv):
        super(SA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)  #Conv2d(40, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_f = conv(f, f, kernel_size=1) #Conv2d(12, 12, kernel_size=(1, 1), stride=(1, 1))
        self.conv_max = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0) #Conv2d(12, 12, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_ = conv(f, f, kernel_size=3, padding=1) #Conv2d(12, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = conv(f, n_feats, kernel_size=1) #Conv2d(12, 40, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = nn.Sigmoid() #Sigmoid()
        self.relu = nn.ReLU(inplace=True) #ReLU(inplace=True)

    def forward(self, x): #输入特性x
        c1_ = (self.conv1(x)) #x通过1×1卷积提取特征
        c1 = self.conv2(c1_) #Conv(S=2) 步长为2的卷积层
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3) #最大池化
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) #双线性插值
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf) #逐元素相加
        m = self.sigmoid(c4) #Sigmoid激活函数
        return x * m #返回x乘m


# 中间的特征提取块------解码block部分--CNN结构
class ExpertBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        in_channels = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # GateWeight
        self.GateWeight = GateWeight()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.conv6 = nn.Conv2d(in_channels=c, out_channels=50, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # 1x1 卷积
        self.conv11 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # 3x3 卷积，为了保持输出尺寸不变，padding 设置为 1
        self.conv33 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 5x5 卷积，为了保持输出尺寸不变，padding 设置为 2
        self.conv55 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)

        self.Expert1 = Expert1(c)    # 专家1
        self.Expert2 = Expert2(c)    # 专家2

        self.t = 3  # t的值为3
        self.K = 3  # K的值为3

        # Gate Network(生成对应权重）
        self.GateNetwork = nn.Sequential(
            nn.Linear(c, c // 4, bias=False),  #线性层   输入：中间特征通道数   输出：中间特征通道数÷4
            nn.ReLU(inplace=True), #激活
            nn.Linear(c // 4, self.K, bias=False), #线性层   输入：中间特征通道数÷4  输出：K=3
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化

    def forward(self, inp):
        x = inp
        a, b, c, d = inp.shape  # 输入的维度
        # 专家1和专家2得到的结果
        Expert1 = self.Expert1(x)
        Expert2 = self.Expert2(x)

        # print(Expert2.shape)

        # 专家3
        x = self.norm1(x)
        conv1_out = self.conv11(x)
        conv3_out = self.conv33(x)
        conv5_out = self.conv55(x)
        x = conv1_out + conv3_out + conv5_out
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.GateWeight(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.GateWeight(x)
        x = self.conv5(x)

        # 专家3得到的结果
        Expert3 = y + x * self.gamma

        # 动态调节机制--生成一组向量
        s = self.avg_pool(inp).view(a, b)  # 对输入的特征进行平均池化
        s = self.GateNetwork(s)  # 进行生成动态调节权值-----输出通道为3
        ax = F.softmax(s / self.t, dim=1)  # 归一化操作（s÷t）---得到两个权值 ax[ : ]
        # print(ax.shape) # ax[ax0  ax1   ax2]相加等于1
        
        
        
        return Expert1 * ax[:, 0].view(a, 1, 1, 1) + Expert2 * ax[:, 1].view(a, 1, 1, 1) + Expert3 * ax[:, 2].view(a, 1,1, 1)
        # return y1 + y  # 加上原特征

######### Multi-scale feature enhancement##################
class MSFEblock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # 第一个卷积层，使用5x5的卷积核和2的填充(padding)。这保持了通道数不变。
        # groups=dim表示每个输入通道都用自己的一组滤波器进行卷积。
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 空间卷积层，使用更大的7x7卷积核和3的膨胀(dilation)。
        # 这在不增加参数数量的情况下增加了感受野。
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 两个卷积层用于将通道维度减半。
        # 这些层分别处理前两个卷积层的输出。
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        # 用于将通道维度从2压缩到2的卷积层，卷积核大小为7。
        # 它用于结合平均和最大注意力机制。
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        # 一个卷积层，将通道维度扩展回其原始大小。
        self.conv = nn.Conv2d(dim // 2, dim, 1)
    def forward(self, x):
        # 应用第一个卷积层
        attn1 = self.conv0(x)
        # 应用空间卷积层
        attn2 = self.conv_spatial(attn1)
        # 减少attn1和attn2的通道维度
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        # 沿着通道维度连接attn1和attn2的输出
        attn = torch.cat([attn1, attn2], dim=1)
        # 计算平均和最大注意力
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # 聚合平均和最大注意力，并应用Sigmoid激活函数
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        # 使用Sigmoid激活加权注意力图并将它们相加
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        # 将通道维度扩展回其原始大小
        attn = self.conv(attn)
        # 用注意力图乘以输入以获得输出
        return x * attn


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V



class FIBlock(nn.Module):
    def __init__(self):
        super(FIBlock, self).__init__()
        self.conv1x1_x = None
        self.conv1x1_y = None
        self.fc_x = None
        self.fc_y = None
        self.conv1x1_concat = None
    def _initialize_layers(self, channels):
        self.conv1x1_x = nn.Conv2d(channels, channels, 1)
        self.conv1x1_y = nn.Conv2d(channels, channels, 1)
        self.fc_x = nn.Linear(channels, channels)
        self.fc_y = nn.Linear(channels, channels)
        self.conv1x1_concat = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x, y, channels):
        if self.conv1x1_x is None or self.conv1x1_x.in_channels != channels:
            self._initialize_layers(channels)

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



# 主干网络
class OURNet24(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        # 输入3通道   宽度16  中间块的数量1  编码器块  解码器块的数量
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)  # 第一个 3X3卷积核
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)  # 最后一个 3X3卷积核

        self.encoders = nn.ModuleList()  # 编码器部分
        self.decoders = nn.ModuleList()  # 解码器部分
        self.middle_blks = nn.ModuleList()  # 中间块
        self.ups = nn.ModuleList()  # 上采样
        self.downs = nn.ModuleList()  # 下采样

        chan = width  # chan=width=16
        #############编码器部分##################
        for num in enc_blk_nums:  # 循环编码器块的数量
            self.encoders.append(
                nn.Sequential(
                    *[ExpertBlock(chan) for _ in range(num)]  # 编码器部分-
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)  # 2X2的卷积核
            )  # 下采样操作
            chan = chan * 2  # 通道数在倍增
        #########中间块部分#######################
        self.middle_blks = \
            nn.Sequential(
                *[ExpertBlock(chan) for _ in range(middle_blk_num)] # 中间块-
            )
        #############解码器部分##################
        for num in dec_blk_nums:  # 解码器的数量
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),  # 1X1卷积
                    nn.PixelShuffle(2)  # 上采样操作
                )
            )
            chan = chan // 2  # 通道数减半
            self.decoders.append(
                nn.Sequential(
                    *[ExpertBlock(chan) for _ in range(num)]  # 解码器部分-----两层的特征增强块---CNN结构
                )
            )
        self.padder_size = 2 ** len(self.encoders)

        ######编解码器交叉注意力############
        # self.FIBlock  = FIBlock()

        ####### 多尺度增强  ####
        self.MSFEblock = MSFEblock(width)


    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)  # 检查输入图片的尺寸
        x = self.intro(inp)  # 第一个3X3卷积提取浅层特征
        x = self.MSFEblock(x)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x) # 输入特征x进入编码器
            encs.append(x)
            x = down(x) # 特征图进行下采样
        x = self.middle_blks(x)  # 中间块

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x) # 特征图上采样
            # channels = x.size(1)  # 获取输入的通道数
            # print("channels", channels)
            # print("编码器特征", enc_skip.shape)
            # e_cda = self.FIBlock(x, enc_skip,channels)  # 输入两个特征--进行交叉注意力
            # print("e_cda", e_cda.shape)

            x = x + enc_skip # 编码器输出的特征和上采样之后的特征相加然后输入解码器
            # print("相加之后的特征", x.shape)
            x = decoder(x)  # 特征图经过解码器
            # x = decoder(e_cda) # 特征图经过解码器
            # print("经过解码之后的特征", x.shape)

        x = self.ending(x) # 最后一个卷积层
        x = x + inp # 输出的结果和原始图像相加

        return x[:, :, :H, :W]

    def check_image_size(self, x):  # 检查图片的尺寸
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# 模型参数计算
def count_parameters(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("total parameters:" + str(k))


class WaterNetLocal(Local_Base, OURNet24):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        OURNet24.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y



