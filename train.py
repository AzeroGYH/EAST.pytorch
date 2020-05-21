import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from dataset import custom_dataset
from model import EAST
from loss import Loss
import os
import time
import numpy as np

from checkpoint import Checkpointer
import argparse
from logger import setup_logger
from utils import get_rank

"""
train_img_path:训练图片路径
train_gt_path:训练图片真值路径
pths_path:模型路径
batch_size:批大小
lr:学习率
num_workers:进程数
epoch_iter:epoch轮数
interval:间隔点
"""
def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval, output_dir):
	# 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(970201)            # 为CPU设置随机种子
    torch.cuda.manual_seed(970201)       # 为当前GPU设置随机种子
	logger = setup_logger("east_matrix", output_dir, get_rank())

	file_num = len(os.listdir(train_img_path)) # 图片数量
	trainset = custom_dataset(train_img_path, train_gt_path) # 训练集进行处理 ？？？ ***
	# 加载数据，组合一个数据集和一个采样器，并在给定的数据集上提供一个可迭代的。
	train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=True, num_workers=num_workers, drop_last=True)
	
	criterion = Loss() # 损失函数 ？？？ ***
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = EAST() # 网络模型 ？？？ ***

	# 是否多gpu
	data_parallel = False 
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		data_parallel = True

	# 分配模型到gpu或cpu，根据device决定
	model.to(device)

	#优化器
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	
	# 学习率衰减策略，一半的时候衰减为十分之一
	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter//2], gamma=0.1)

	# 判断是否有保存的模型，有的话加载最后一个继续训练
    checkpointer = Checkpointer(
        model, optimizer, scheduler, pths_path
    )
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    start_epoch = arguments['iteration'] # 开始的轮数

    logger.info('start_epoch is :{}'.format(start_epoch))

	for epoch in range(start_epoch, epoch_iter):
		iteration = epoch + 1
        arguments['iteration'] = iteration	
		model.train()
		epoch_loss = 0 # 初始化每一轮的损失为0
		epoch_time = time.time() # 记录每一轮的时间
		for i, (img, gt_score, gt_geo, ignored_map) in enumerate(train_loader):
			start_time = time.time() # 记录每一个batch的时间
			img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(device), gt_geo.to(device), ignored_map.to(device)
			# 将图片输入模型
			pred_score, pred_geo = model(img)
			# 计算损失loss
			loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
			
			epoch_loss += loss.item()
			optimizer.zero_grad() # 优化器梯度归零
			loss.backward() # 梯度反传
			optimizer.step() # 梯度更新
	
			scheduler.step()


			print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format(\
              epoch+1, epoch_iter, i+1, int(file_num/batch_size), time.time()-start_time, loss.item()))
		
		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/int(file_num/batch_size), time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)
		# 判断是否需要保存模型
        if iteration % interval == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == epoch_iter:
            checkpointer.save("model_final", **arguments)


if __name__ == '__main__':
	torch.multiprocessing.set_sharing_strategy('file_system')

	train_img_path = os.path.abspath('../ICDAR_2015/train_img')
	train_gt_path  = os.path.abspath('../ICDAR_2015/train_gt')
	pths_path      = './pths'
	output_dir     = './log'
	batch_size     = 24 
	lr             = 1e-3
	num_workers    = 4
	epoch_iter     = 1000
	save_interval  = 50
	train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval, output_dir)	
	
