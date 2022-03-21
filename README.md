# CNN
基于pytorch各种CNN基础模型复现 

### Requirements
  `torch`  
  `torchvision`  
  `matplotlib`  

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
探索更深层次的网络结构，用小卷积核代替大卷积核，减少参数  
[Paper](https://arxiv.org/pdf/1409.1556.pdf) 

2014 -- GoogLeNet v1  
引入inception，在流行“加深”的时候提供了“加宽”的思路，采用平均池化代替全连接层的思想  
引入前置1x1卷积减少参数量（先用1x1卷积降低深度，再进行其他卷积）  
[Paper](https://arxiv.org/pdf/1409.4842.pdf)  
......

