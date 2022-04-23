### LSTM
[Paper](http://www.bioinf.jku.at/publications/older/2604.pdf)  
[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)  
Dataset: Bitcoin daily prices, U.S. dollars per bitcoin.([Source](https://www.nasdaq.com/)) 

#### Architecture
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/lstm.png'height=250>
</div>

#### Gate of LSTM
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/lstm_parts.png'height=450>
</div>

#### torch
```
torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True,    
              batch_first=False, dropout=0, bidirectional=False, proj_size=0)
```
```
input:  [seq_len, batch_size, input_size]  
output: [seq_len, batch_size, hidden_size]
```
batch_size在第二维，与卷积网络中不太相同，这是因为LSTM Unit是一个token接着一个token输入的，结合下图理解，总共seq_len个token（本次输入需要等到上一次输出后），一个token有input_size维。有batch_size个token数量为seq_len的sequence同时输入，LSTM单元每次输入batch_size个token（来自不同sequence）。
<div align='center'>
  <img src='https://github.com/Luxlios/Figure/blob/main/CNN/lstm_fun.png'height=350>
</div>
