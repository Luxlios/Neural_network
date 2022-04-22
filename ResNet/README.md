### ResNet
[Paper](https://arxiv.org/pdf/1512.03385.pdf)  
Dataset: [ImageNet](https://image-net.org/)

#### Block  
With the network depth increasing, accuracy gets saturated and then degrades rapidly.   
在这个问题的基础上可认为网络难以学习identity mapping，构建shortcut connection 相当于identity mapping，作为一个先验知识引导网络学习，如图左边Block所示。为了进一步研究更加深层次的网络，同时需要避免参数量暴增，降低计算量，借助Inception v1的思想，通过1×1卷积降低计算量，如图右边Block所示(Bottleneck architecture)。  
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/resblock.png'height=300>
</div>
左边用于ResNet-18, ResNet-34，右边用于ResNet-50, ResNet-101, ResNet-152   

#### Architecture
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/resnet.png'height=300>
</div>
Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.
