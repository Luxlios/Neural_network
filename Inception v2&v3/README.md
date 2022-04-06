## Inception v2&v3  
[Paper](https://arxiv.org/pdf/1512.00567v3.pdf)

Dataset:[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)  

### Content
- [Batch Normalization](#batch-normalization)
- [New Inception Architecture](#new-inception-architecture)
  * [Inception A](#inception-a)
  * [Inception B](#inception-b)
  * [Inception C](#inception-c)
  * [Inception D](#inception-d)
  * [Auxiliary classifier](#auxiliary-classifier)
- [Inception v2 Architecture](#inception-v2-architecture)
- [Inception v3 Architecture](#inception-v3-architecture)

### Batch Normalization
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/bn.png' height='300'>
</div>

```
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

### New Inception Architecture
#### Inception A 
##### 用两个3×3卷积代替5×5卷积
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_a.png' height='300'>
</div>

#### Inception B
##### 用1×n与n×1卷积级联代替n×n卷积
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_b.png' height='400'>
</div>

#### Inception C
##### 提高通道层数
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_c.png' height='300'>
</div>

#### Inception D
##### 降低高度与宽度的同时提高通道层数
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_d.png' height='300'>
</div>

#### Auxiliary classifier
##### locate in top of the last 17×17 layer
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/auxiliary_classifier.png' height='300'>
</div>

### Inception v2 Architecture
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/Inception_v2.png' height='400'>
</div>

### Inception v3 Architecture
在Inception v2架构的基础上添加几个点：  
- Optimizer: RMSProp
1. with decay of 0.9 and epsilon=1.0  
2. learning rate of 0.045, decayed every two epoch using an exponential rate of 0.94  
3. gradient clipping with threshold 2.0(be found to be useful to stabilize the training)  
- Label Smoothing Regularization
- Factorize the first 7×7 convolutional layer into a sequence of 3×3 convolutional layers
- Auxiliary classifier + BN
