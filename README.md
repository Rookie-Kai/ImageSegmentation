# Image Segmentation

本项目基于OpenMMLab图像分割工具包 MMSegmentation进行构建，使用前请先配置好MMSegmentation工作环境

###  环境配置

1. 安装torch

   <code>pip install torch torchvision torchaudio</code>

2. mim安装mmcv

   ```
   pip install -U openmim
   mim install mmengine
   mim install 'mmcv==2.0.0rc3'
   ```

   注：若后续提示<code>Please install mmcv>=2.0.0rc4.</code>，运行<code>mim install 'mmcv==2.0.0rc4'</code>即可，mim会自动卸载rc3版本更换为rc4

3. 安装其他工具包

   <code>pip install opencv-python pillow matplotlib seaborn tqdm 'mmdet>=3.0.0rc1' -i https://pypi.tuna.tsinghua.edu.cn/simple</code>

4. 下载并安装MMSegmentation

   ```
   git clone https://github.com/open-mmlab/mmsegmentation.git -b dev-1.x
   cd mmsegmentation
   pip install -v -e .
   ```

5. 下载数据集和预训练权重

   ```
   # 数据集
   wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/Glomeruli-dataset.zip
   
   # 预训练权重
   wget https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth -P
   
   # resnet50_v1c
   wget https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth
   ```



### 模型训练与推理

1. 模型配置文件

   通过<code>bash/run.sh</code>脚本运行<code>train_cfg.py</code>进行模型训练，同时生成配置文件<code>pspnet_glom.py</code>

   train_log文件见<code>log/slurm-515750.out</code>和<code>log/20230213_102904/</code>

   |   Class    |  IoU  |  Acc  |
   | :--------: | :---: | :---: |
   | background | 99.42 | 99.73 |
   | glomeruili | 74.17 | 84.29 |

2. 训练好的模型权重

   链接：https://pan.baidu.com/s/1zKHpxHvgpVmMM8BI-2v9ZA 
   提取码：hlb3

3. 在测试集上进行推理

   运行<code>bash/test.sh</code>脚本，在测试集上进行推理

   test_log文件见<code>log/slurm-515769.out</code>和<code>log/20230213_104653/</code>

   推理结果如下图所示

   ![](https://github.com/Rookie-Kai/ImageSegmentation/blob/main/data/test_SAS_21883_001_62.png_0.png?raw=true)

   ![](https://github.com/Rookie-Kai/ImageSegmentation/blob/main/data/test_VUHSK_1762_18.png_0.png?raw=true)

4. 速度指标

   运行<code>bash/benchmark.sh</code>脚本，查看模型的推理速度（FPS）

   log文件见<code>log/slurm-515772.out</code>和<code>log/20230213_105145/</code>

   结果如下：

   > Done image [50 / 200], fps: 72.77 img / s
   >  Done image [100/ 200], fps: 73.04 img / s
   >  Done image [150/ 200], fps: 73.21 img / s
   >  Done image [200/ 200], fps: 73.27 img / s
   >  Overall fps: 73.27 img / s
   >
   >  Average fps of 1 evaluations: 73.27
   >  The variance of 1 evaluations: 0.0

