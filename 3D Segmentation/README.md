### 3D Segmentation
[3D U-Net](https://arxiv.org/pdf/1606.06650.pdf)&nbsp;&nbsp;&nbsp;&nbsp;
[KiU-Net](https://arxiv.org/pdf/2006.04878.pdf)  
Dataset: 

### Architecture
#### 3D U-Net
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/3dunet.png'height=350>
</div>

#### KiU-Net
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN1/kinetunit.png'height=150>
</div>
U-Net逐步降采样增加感受野，如(a)，同时也会降低网络对于细节学习的能力，这对于分割任务是一个天然的缺陷。KiU-Net中提出一个Ki-Net架构，基本单元如(b)，通过上采样使得网络用原图成倍的参数表示原图，提高网络对于细节的学习能力。
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/2dkiunet.png'height=400>
</div>
通过CRFB模块（类似残差块）将U-Net与Ki-Net结合在一起，融合U-Net较好的分割能力和Ki-Net对于细节的处理能力，提出KiU-Net架构。上图所示为2d的KiU-Net架构，直接将2d的输入输出改为3d即为3d的KiU-Net架构。需要注意的是，对于3d的KiU-Net架构，由于Ki-Net的上采样，参数量爆炸增长，对于通道数的设置不能过大。
