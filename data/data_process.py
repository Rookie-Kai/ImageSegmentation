import os
import random


image_path = '/data/home/scv9243/run/mmsegmentation/data/Glomeruli-dataset/images'

all_image_list = os.listdir(image_path)
all_image_num = len(all_image_list)

# 打乱顺序
random.shuffle(all_image_list)

# 划分训练集和测试集
train_ratio = 0.8
test_ratio = 1 - train_ratio

train_image_list = all_image_list[:int(all_image_num * train_ratio)]
train_image_num = len(train_image_list)
test_image_list = all_image_list[int(all_image_num * train_ratio):]
test_image_num = len(test_image_list)

print('数据集图像总数', all_image_num)
print('训练集划分比例', train_ratio)
print('训练集图像个数', train_image_num)
print('测试集图像个数', test_image_num)


with open('/data/home/scv9243/run/mmsegmentation/data/Glomeruli-dataset/splits/train.txt', 'w', encoding='utf-8') as f:
    f.writelines(line.split('.')[0] + '\n' for line in train_image_list)
with open('/data/home/scv9243/run/mmsegmentation/data/Glomeruli-dataset/splits/val.txt', 'w', encoding='utf-8') as f:
    f.writelines(line.split('.')[0] + '\n' for line in test_image_list)


