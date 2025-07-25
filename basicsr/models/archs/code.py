######### 我们设计的MSFEblock---多尺度特征增强模块Multi-scale feature enhancement##################
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
