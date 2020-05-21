import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math

# vgg16 block1~5的各个卷积层的输出通道数
cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


'''
VGG16卷积层
'''
def make_layers(cfg, batch_norm=False): # vgg16没有bn层，vgg16_bn则有bn层
	layers = []
	in_channels = 3 # 输入通道为3，图片的三通道
	for v in cfg:
		if v == 'M': # 添加池化层，卷积核大小为2，步长为2
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		else:
			# 3*3的卷积层，padding为1
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers) # 模块序列化容器


class VGG(nn.Module):
	def __init__(self, features):
		super(VGG, self).__init__()
		self.features = features # 特征提取层
		self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) # 为了与全连接层的输出对应
		# 后面的三个全连接层
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1000),
		)

		# 参数初始化
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class extractor(nn.Module):
	def __init__(self, pretrained):
		super(extractor, self).__init__()
		# 首先，构建VGG16的结构
		vgg16_bn = VGG(make_layers(cfg, batch_norm=True))
		# 如果有预训练好的模型，则加载参数model.load_state_dict(torch.load(model_path))
		if pretrained:
			vgg16_bn.load_state_dict(torch.load('./pths/vgg16_bn-6c64b313.pth'))
		self.features = vgg16_bn.features # 返回预训练模型初始化的特提取层
	
	def forward(self, x):
		out = []
		for m in self.features:
			x = m(x)
			# 是否为block的结束，结束需保存，后面的特征融合需要用到
			if isinstance(m, nn.MaxPool2d):
				out.append(x)
		return out[1:] # 返回block2~5的进行使用


class merge(nn.Module):
	def __init__(self):
		super(merge, self).__init__()

		self.conv1 = nn.Conv2d(1024, 128, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.relu1 = nn.ReLU()
		self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
		self.bn2 = nn.BatchNorm2d(128)
		self.relu2 = nn.ReLU()

		self.conv3 = nn.Conv2d(384, 64, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.relu3 = nn.ReLU()
		self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.relu4 = nn.ReLU()

		self.conv5 = nn.Conv2d(192, 32, 1)
		self.bn5 = nn.BatchNorm2d(32)
		self.relu5 = nn.ReLU()
		self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn6 = nn.BatchNorm2d(32)
		self.relu6 = nn.ReLU()

		self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
		self.bn7 = nn.BatchNorm2d(32)
		self.relu7 = nn.ReLU()
		
		# 参数初始化
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		# 上采样以便与特征提取的上一层特征concatnate
		y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True) 
		y = torch.cat((y, x[2]), 1)
		# 对concatnate之后的特征执行两个卷积
		y = self.relu1(self.bn1(self.conv1(y)))		
		y = self.relu2(self.bn2(self.conv2(y)))
		
		# 以下同上操作
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[1]), 1)
		y = self.relu3(self.bn3(self.conv3(y)))		
		y = self.relu4(self.bn4(self.conv4(y)))
		
		y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
		y = torch.cat((y, x[0]), 1)
		y = self.relu5(self.bn5(self.conv5(y)))		
		y = self.relu6(self.bn6(self.conv6(y)))
		
		# 最后一次不为融合，只是再进行一次卷积操作
		y = self.relu7(self.bn7(self.conv7(y)))
		return y

'''
实现EAST中的RROX方式输出：
	score: 分类分数
	geo: 形状预测，包括位置loc和角度预测angle
'''
class output(nn.Module):
	def __init__(self, scope=512):
		super(output, self).__init__()
		# 1*1的卷积核，输入通道数为32，输出通道数为1，得到预测分数
		self.conv1 = nn.Conv2d(32, 1, 1)
		self.sigmoid1 = nn.Sigmoid()

		# 1*1的卷积核，输入通道数为32，输出通道数为4，得到像素到四个边界的距离预测
		self.conv2 = nn.Conv2d(32, 4, 1)
		self.sigmoid2 = nn.Sigmoid()

		# 1*1的卷积核，输入通道数为32，输出通道数为1，得到预测的角度
		self.conv3 = nn.Conv2d(32, 1, 1)
		self.sigmoid3 = nn.Sigmoid()
		self.scope = 512 # 原图的大小

		# 参数初始化
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		score = self.sigmoid1(self.conv1(x))
		loc   = self.sigmoid2(self.conv2(x)) * self.scope
		angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi # angle is between [-90, 90]
		geo   = torch.cat((loc, angle), 1) # 按列拼接位置和角度预测到一个总的形状预测中
		return score, geo
		
	
class EAST(nn.Module):
	def __init__(self, pretrained=True):
		super(EAST, self).__init__()
		# 提取预训练模型的特征提取层
		self.extractor = extractor(pretrained)
		# 特征合并
		self.merge = merge()
		# 模型的输出
		self.output = output()
	
	def forward(self, x):
		return self.output(self.merge(self.extractor(x)))
		
if __name__ == '__main__':
	m = EAST()
	x = torch.randn(1, 3, 256, 256)
	score, geo = m(x)
	print(score.shape)
	print(geo.shape)
