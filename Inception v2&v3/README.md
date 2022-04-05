## Inception v2&v3  
[Paper](https://arxiv.org/pdf/1512.00567v3.pdf)

Dataset:[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)  

### Content
- [Batch Normalization](#Batch Normalization)
- [New Inception Architecture](#New Inception Architecture)
  - [Inception A](#Inception A)
  - [Inception B](#Inception B)
  - [Inception C](#Inception C)
  - [Inception D](#Inception D)

### Batch Normalization
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/bn.png' height='300'>
</div>

### New Inception Architecture
#### Inception A 
##### 用两个3×3卷积代替5×5卷积
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_a.png' height='300'>
</div>

#### Inception B
##### 用1×n与n×1卷积级联代替n×n卷积
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_b.png' height='300'>
</div>

#### Inception C
##### 提高通道层数
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_c.png' height='300'>
</div>

#### Inception D
#### 降低高宽度的同时提高通道层数
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/inception_d.png' height='300'>
</div>

