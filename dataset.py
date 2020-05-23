from shapely.geometry import Polygon
import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import random


def cal_distance(x1, y1, x2, y2):
	'''calculate the Euclidean distance'''
	return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def move_points(vertices, index1, index2, r, coef):
	'''move the two points to shrink edge 移动两点以收缩一条边
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
		index1  : offset of point1
		index2  : offset of point2
		r       : [r1, r2, r3, r4] in paper
		coef    : shrink ratio in paper
	Output:
		vertices: vertices where one edge has been shinked
	'''
	index1 = index1 % 4
	index2 = index2 % 4
	x1_index = index1 * 2 + 0
	y1_index = index1 * 2 + 1
	x2_index = index2 * 2 + 0
	y2_index = index2 * 2 + 1
	
	r1 = r[index1]
	r2 = r[index2]
	length_x = vertices[x1_index] - vertices[x2_index]
	length_y = vertices[y1_index] - vertices[y2_index]
	length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])
	if length > 1:	
		ratio = (r1 * coef) / length
		vertices[x1_index] += ratio * (-length_x) 
		vertices[y1_index] += ratio * (-length_y) 
		ratio = (r2 * coef) / length
		vertices[x2_index] += ratio * length_x 
		vertices[y2_index] += ratio * length_y
	return vertices	


def shrink_poly(vertices, coef=0.3): 
	'''shrink the text region 收缩文本区域得到score map
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
		coef    : shrink ratio in paper
	Output:
		v       : vertices of shrinked text region <numpy.ndarray, (8,)>
	'''
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	r1 = min(cal_distance(x1,y1,x2,y2), cal_distance(x1,y1,x4,y4))
	r2 = min(cal_distance(x2,y2,x1,y1), cal_distance(x2,y2,x3,y3))
	r3 = min(cal_distance(x3,y3,x2,y2), cal_distance(x3,y3,x4,y4))
	r4 = min(cal_distance(x4,y4,x1,y1), cal_distance(x4,y4,x3,y3))
	r = [r1, r2, r3, r4]

	# obtain offset to perform move_points() automatically
	if cal_distance(x1,y1,x2,y2) + cal_distance(x3,y3,x4,y4) > \
       cal_distance(x2,y2,x3,y3) + cal_distance(x1,y1,x4,y4):
		offset = 0 # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
	else:
		offset = 1 # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

	v = vertices.copy()
	v = move_points(v, 0 + offset, 1 + offset, r, coef)
	v = move_points(v, 2 + offset, 3 + offset, r, coef)
	v = move_points(v, 1 + offset, 2 + offset, r, coef)
	v = move_points(v, 3 + offset, 4 + offset, r, coef)
	return v


def get_rotate_mat(theta): # 角度对应的正余弦值
	'''positive theta value means rotate clockwise'''
	return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None): 
	'''rotate vertices around anchor 围绕anchor点旋转各个顶点，也就是旋转矩形
	Input:	
		vertices: vertices of text region <numpy.ndarray, (8,)>
		theta   : angle in radian measure
		anchor  : fixed position during rotation
	Output:
		rotated vertices <numpy.ndarray, (8,)>
	'''
	v = vertices.reshape((4,2)).T
	if anchor is None: # 不指定则默认第一个点，左上角
		anchor = v[:,:1]
	rotate_mat = get_rotate_mat(theta)
	res = np.dot(rotate_mat, v - anchor)
	return (res + anchor).T.reshape(-1)


def get_boundary(vertices): 
	'''get the tight boundary around given vertices 给定点四边形的最小外接矩形
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		the boundary
	'''
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	x_min = min(x1, x2, x3, x4)
	x_max = max(x1, x2, x3, x4)
	y_min = min(y1, y2, y3, y4)
	y_max = max(y1, y2, y3, y4)
	return x_min, x_max, y_min, y_max


def cal_error(vertices): 
	'''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
	calculate the difference between the vertices orientation and default orientation
	寻找水平矩形旋转最好的角度，以便和真正的四边形真值框有最小误差
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		err     : difference measure
	'''
	x_min, x_max, y_min, y_max = get_boundary(vertices)
	x1, y1, x2, y2, x3, y3, x4, y4 = vertices
	err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
	return err	


def find_min_rect_angle(vertices):
	'''find the best angle to rotate poly and obtain min rectangle
	寻找最好的旋转角度，方法是一度一度旋转去计算最好的
	Input:
		vertices: vertices of text region <numpy.ndarray, (8,)>
	Output:
		the best angle <radian measure>
	'''
	angle_interval = 1
	angle_list = list(range(-90, 90, angle_interval)) # 角度范围为[-90,90]
	area_list = []
	for theta in angle_list: 
		rotated = rotate_vertices(vertices, theta / 180 * math.pi)
		x1, y1, x2, y2, x3, y3, x4, y4 = rotated
		temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
		area_list.append(temp_area)
	
	sorted_area_index = sorted(list(range(len(area_list))), key=lambda k : area_list[k])
	min_error = float('inf')
	best_index = -1
	rank_num = 10
	# find the best angle with correct orientation
	for index in sorted_area_index[:rank_num]:
		rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
		temp_error = cal_error(rotated)
		if temp_error < min_error:
			min_error = temp_error
			best_index = index
	return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
	'''check if the crop image crosses text regions
	Input:
		start_loc: left-top position
		length   : length of crop image
		vertices : vertices of text regions <numpy.ndarray, (n,8)>
	Output:
		True if crop image crosses text region
	'''
	if vertices.size == 0:
		return False
	start_w, start_h = start_loc
	a = np.array([start_w, start_h, start_w + length, start_h, \
          start_w + length, start_h + length, start_w, start_h + length]).reshape((4,2))
	p1 = Polygon(a).convex_hull
	for vertice in vertices:
		p2 = Polygon(vertice.reshape((4,2))).convex_hull
		inter = p1.intersection(p2).area
		if 0.01 <= inter / p2.area <= 0.99: 
			return True
	return False
		

def crop_img(img, vertices, labels, length):
	'''crop img patches to obtain batch and augment
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
		length      : length of cropped image region
	Output:
		region      : cropped image region
		new_vertices: new vertices in cropped region
	'''
	h, w = img.height, img.width
	# confirm the shortest side of image >= length
	if h >= w and w < length:
		img = img.resize((length, int(h * length / w)), Image.BILINEAR)
	elif h < w and h < length:
		img = img.resize((int(w * length / h), length), Image.BILINEAR)
	ratio_w = img.width / w
	ratio_h = img.height / h
	assert(ratio_w >= 1 and ratio_h >= 1)

	new_vertices = np.zeros(vertices.shape)
	if vertices.size > 0:
		new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
		new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h

		# find random position
		remain_h = img.height - length
		remain_w = img.width - length
		flag = True
		cnt = 0
		while flag and cnt < 1000:
			cnt += 1
			start_w = int(np.random.rand() * remain_w)
			start_h = int(np.random.rand() * remain_h)
			flag = is_cross_text([start_w, start_h], length, new_vertices[labels==1,:])
		box = (start_w, start_h, start_w + length, start_h + length)
		region = img.crop(box)
		if new_vertices.size == 0:
			return region, new_vertices	
		
		new_vertices[:,[0,2,4,6]] -= start_w
		new_vertices[:,[1,3,5,7]] -= start_h
		return region, new_vertices
	else:
		# find random position
		remain_h = img.height - length
		remain_w = img.width - length
		start_w = int(np.random.rand() * remain_w)
		start_h = int(np.random.rand() * remain_h)
		box = (start_w, start_h, start_w + length, start_h + length)
		region = img.crop(box)
		if new_vertices.size == 0:
			return region, new_vertices	
		
		new_vertices[:,[0,2,4,6]] -= start_w
		new_vertices[:,[1,3,5,7]] -= start_h
		return region, new_vertices
		

def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
	'''get rotated locations of all pixels for next stages
	Input:
		rotate_mat: rotatation matrix
		anchor_x  : fixed x position
		anchor_y  : fixed y position
		length    : length of image
	Output:
		rotated_x : rotated x positions <numpy.ndarray, (length,length)>
		rotated_y : rotated y positions <numpy.ndarray, (length,length)>
	'''
	x = np.arange(length)
	y = np.arange(length)
	x, y = np.meshgrid(x, y)
	x_lin = x.reshape((1, x.size))
	y_lin = y.reshape((1, x.size))
	coord_mat = np.concatenate((x_lin, y_lin), 0)
	rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                                                   np.array([[anchor_x], [anchor_y]])
	rotated_x = rotated_coord[0, :].reshape(x.shape)
	rotated_y = rotated_coord[1, :].reshape(y.shape)
	return rotated_x, rotated_y


def adjust_height(img, vertices, ratio=0.2):
	'''adjust height of image to aug data 调整图片高度以图片增强
	Input:
		img         : PIL Image 图片
		vertices    : vertices of text regions <numpy.ndarray, (n,8)> 顶点
		ratio       : height changes in [0.8, 1.2] 高度变化范围
	Output:
		img         : adjusted PIL Image 调整后的图片
		new_vertices: adjusted vertices 调整后的顶点
	'''
	ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
	old_h = img.height
	new_h = int(np.around(old_h * ratio_h))
	img = img.resize((img.width, new_h), Image.BILINEAR) # 缩放图片
	
	new_vertices = vertices.copy()
	if vertices.size > 0:
		new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * (new_h / old_h)
	return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
	'''rotate image [-10, 10] degree to aug data 数据增强：旋转角度[-10,10]
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		angle_range : rotate range
	Output:
		img         : rotated PIL Image
		new_vertices: rotated vertices
	'''

	# 图片中心点
	center_x = (img.width - 1) / 2
	center_y = (img.height - 1) / 2

	# 随机旋转角度
	angle = angle_range * (np.random.rand() * 2 - 1)
	img = img.rotate(angle, Image.BILINEAR)

	#计算新的顶点坐标
	new_vertices = np.zeros(vertices.shape)
	for i, vertice in enumerate(vertices):
		new_vertices[i,:] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x],[center_y]]))
	return img, new_vertices

def rotate_img_random(img, vertices):
	'''rotate image [0,90,180,270] degree to aug data 数据增强:随机旋转0，90，180，270
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		angle_range : rotate range
	Output:
		img         : rotated PIL Image
		new_vertices: rotated vertices
	'''
	angles_op = [0,Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270]
	angle = random.choice([0,Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270])
	if angle == 0:
		return img, vertices
	else:
		# 图片旋转
		img = img.transpose(angle)
		w = img.width
		h = img.height
		new_vertices = np.zeros(vertices.shape)
		# 坐标变换
		if angle == 2: # Image.ROTATE_90 anticlockwise
			for i, vertice in enumerate(vertices):
				temp_vertice = vertice
				for k in range(0,len(temp_vertice),2):
					old_x = temp_vertice[k]
					temp_vertice[k] = temp_vertice[k+1]
					temp_vertice[k+1] = w - old_x
				new_vertices[i,:] = temp_vertice

		elif angle == 3 : # Image.ROTATE_180
			for i, vertice in enumerate(vertices):
				temp_vertice = vertice
				for k in range(0,len(temp_vertice),2):
					old_x = temp_vertice[k]
					old_y = temp_vertice[k+1]
					temp_vertice[k] = w - old_x
					temp_vertice[k+1] = h - old_y
				new_vertices[i,:] = temp_vertice
		else: # Image.ROTATE_270
			for i, vertice in enumerate(vertices):
				temp_vertice = vertice
				for k in range(0,len(temp_vertice),2):
					old_y = temp_vertice[k+1]
					temp_vertice[k+1] = temp_vertice[k]
					temp_vertice[k] = h - old_y
				new_vertices[i,:] = temp_vertice
		
		return img, new_vertices


def get_score_geo(img, vertices, labels, scale, length):
	'''generate score gt and geometry gt 生成分数真值和形状真值
	Input:
		img     : PIL Image
		vertices: vertices of text regions <numpy.ndarray, (n,8)>
		labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
		scale   : feature map / image
		length  : image length
	Output:
		score gt, geo gt, ignored
	'''
	# 初始化数组维度，特征图的长*宽，score为1通道，geo为5通道
	score_map   = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
	geo_map     = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)
	ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)
	
	# 获取特征图点对应原图的坐标点
	index = np.arange(0, length, int(1/scale))
	index_x, index_y = np.meshgrid(index, index)

	ignored_polys = [] # 无法识别的文字区域
	polys = [] # 文字区域
	
	for i, vertice in enumerate(vertices):
		if labels[i] == 0: 
			ignored_polys.append(np.around(scale * vertice.reshape((4,2))).astype(np.int32))
			continue		
		
		poly = np.around(scale * shrink_poly(vertice).reshape((4,2))).astype(np.int32) # scaled & shrinked
		polys.append(poly)
		temp_mask = np.zeros(score_map.shape[:-1], np.float32)
		cv2.fillPoly(temp_mask, [poly], 1)
		
		theta = find_min_rect_angle(vertice)
		rotate_mat = get_rotate_mat(theta)
		
		rotated_vertices = rotate_vertices(vertice, theta)
		x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
		rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)
	
		d1 = rotated_y - y_min
		d1[d1<0] = 0
		d2 = y_max - rotated_y
		d2[d2<0] = 0
		d3 = rotated_x - x_min
		d3[d3<0] = 0
		d4 = x_max - rotated_x
		d4[d4<0] = 0
		geo_map[:,:,0] += d1[index_y, index_x] * temp_mask
		geo_map[:,:,1] += d2[index_y, index_x] * temp_mask
		geo_map[:,:,2] += d3[index_y, index_x] * temp_mask
		geo_map[:,:,3] += d4[index_y, index_x] * temp_mask
		geo_map[:,:,4] += theta * temp_mask
	
	cv2.fillPoly(ignored_map, ignored_polys, 1)
	cv2.fillPoly(score_map, polys, 1)
	return torch.Tensor(score_map).permute(2,0,1), torch.Tensor(geo_map).permute(2,0,1), torch.Tensor(ignored_map).permute(2,0,1)


def extract_vertices(lines):
	'''extract vertices info from txt lines从txt文本行获取顶点信息
	Input:
		lines   : list of string info 文本行list集合
	Output:
		vertices: vertices of text regions <numpy.ndarray, (n,8)> 文字区域的顶点
		labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)> 是否有效
	'''
	labels = []
	vertices = []
	for line in lines:
		vertices.append(list(map(int,line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
		label = 0 if '###' in line else 1
		labels.append(label)
	return np.array(vertices), np.array(labels)

"""
属性：
	img_path：训练集图片的路径
	gt_path: 训练集真值的路径
	scale: feature map / image 特征图相对原图的比例为1/4
	length: crop的大小
"""
class custom_dataset(data.Dataset):
	def __init__(self, img_path, gt_path, scale=0.25, length=512):
		super(custom_dataset, self).__init__()
		# 获取参数
		self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
		self.gt_files  = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
		self.scale = scale
		self.length = length

	def __len__(self): # 返回整个数据集的长度
		return len(self.img_files)

	def __getitem__(self, index): # 定义每次怎么读取数据，支持获取给定键的数据样本

		# 读取真值框信息
		with open(self.gt_files[index], 'r') as f:
			lines = f.readlines()
		vertices, labels = extract_vertices(lines)
		
		# 图片增强处理
		img = Image.open(self.img_files[index]) # 打开对应图片
		img, vertices = adjust_height(img, vertices) # 调整高度
		img, vertices = crop_img(img, vertices, labels, self.length) # crop图片

		img, vertices = rotate_img_random(img, vertices) # 旋转角度
		
		transform = transforms.Compose([transforms.ColorJitter(0.5, 0.5, 0.5, 0.25), \
                                        transforms.ToTensor(), \
                                        transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
		
		# 获取对应的三个map
		score_map, geo_map, ignored_map = get_score_geo(img, vertices, labels, self.scale, self.length)
		return transform(img), score_map, geo_map, ignored_map

