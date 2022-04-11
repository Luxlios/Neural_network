## U-Net
[Paper](https://arxiv.org/pdf/1505.04597.pdf)  
Dataset: [TGS Salt Identification Challenge](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data)

### Architecture
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/U-Net.png'height='400'>
</div>

### Strategy(Symmetric padding)
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/UNet_symmetric.png'height='250'>
</div>
  使用该策略避免图片边缘像素没有分割信息，同时，从Architecture中可以看到，图像输入的尺寸为572×572， 输出的尺寸为388×388，在copy and crop环节直接采用裁剪的方式进行连接，会导致边缘信息的丢失，使用该策略，将图像输入尺寸置为388×388，通过symmetric补充92像素，得到572×572再输入到网络中训练，这样在copy and crop环节能够减少边缘部分信息的丢失。
