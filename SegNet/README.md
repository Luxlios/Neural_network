### SegNet
[Paper](https://arxiv.org/pdf/1511.00561.pdf)  
Dataset: [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)[(Download)](https://s3.amazonaws.com/fast-ai-imagelocal/camvid.tgz)  

### Architecture
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/segnet.png'height=250>
</div>
论文引入一个新指标boundary F1-measure(BF)来衡量boundary accuracy。同时提出一个基于VGG16的架构SegNet，相比于FCN，采用encoder中的pooling indices直接进行上采样，不用像FCN一样保存对应encoder的整个feature map，节省训练消耗；在decoder中增加与encoder相对称的conv层用来平滑由于maxunpool带来的sparse feature map，虽然增大了inference的消耗，但是分割的效果显著提升，轮廓效果更加好。
