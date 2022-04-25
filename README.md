### Neural Network
基于pytorch各种神经网络基础模型复现 

### Requirements
  `torch`  
  `torchvision`  
  `matplotlib`  
  `sklearn`

### CNN模型的发展
1986 -- MLP  
引入back propagation  
[Paper](http://www.cs.toronto.edu/~bonner/courses/2016s/csc321/readings/Learning%20representations%20by%20back-propagating%20errors.pdf)  

1998 -- LeNet-5  
引入convolution和pooling  
[Paper](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Convolution_nets.pdf)  

2012 -- AlexNet  
向深度发展，引入dropout, data augmentation和local response normalization  
[Paper](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)  

2014 -- VGG  
探索更深层次的网络结构，用多个小卷积核代替大卷积核（感受野不变），减少参数  
[Paper](https://arxiv.org/pdf/1409.1556.pdf) 

2014 -- Inception v1(GoogLeNet)  
引入inception，在流行“加深”的时候提供了“加宽”的思路  
引入前置1x1卷积减少参数量（先用1x1卷积降低深度，再进行其他卷积）  
中间两个部位加入softmax层，防止梯度消失，加速网络收敛， 最终预测时这两个部分需要去掉  
采用平均池化代替全连接层的思想，最后依旧加了一个全连接层方便finetune    
[Paper](https://arxiv.org/pdf/1409.4842.pdf)  

2015 -- Inception v2&v3  
引入Batch Normalization  
[Paper](https://arxiv.org/pdf/1502.03167v3.pdf)   
在BN与Inception结合的基础上提出了Inception v2&v3  
1.学习VGG，采用两个3x3卷积代替5x5卷积  
2.将nxn卷积分别为1xn卷积和nx1卷积，减少1/3的cost  
3.GooLeNet的额外分支加入BN，将它们看作为正则项  
4.inception“更宽”消除表示瓶颈（用并行的形式代替串行，减少计算量的同时不会降低表征能力）  
5.引入LSR(Label-Smoothing Regularization)处理标记，减少过拟合以及对于数据的信赖  
6.采用RMSProp优化器进行训练  
7.Inception v1中第一层的7×7卷积也采用1进行分解    
Inception v2中采用了技巧1、2、4和7  
Inception v3中采用了全部的技巧（1-7）  
[Paper](https://arxiv.org/pdf/1512.00567v3.pdf)   
[网络Optimizer参考Paper](https://arxiv.org/pdf/1609.04747.pdf)  

2015 -- ResNet  
[Paper](https://arxiv.org/pdf/1512.03385.pdf)  
引入两种residual learning的架构，解决由于网络加深发生退化的问题  

2016 -- Inception v4  
[Paper](https://arxiv.org/pdf/1602.07261.pdf)  
......

