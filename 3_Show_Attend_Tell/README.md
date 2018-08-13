# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
https://arxiv.org/pdf/1502.03044.pdf


Here are some comments:
1. Here 'Deterministic "Soft" Attention' is implemented. (Not 'Stochastic "Hard" Attention)
2. Encoder: I used pretrained VGG19, the last CNN layer to extract features (14x14x512)
3. Decoder: RNN with Soft Attention
4. I refer to many of codes from https://github.com/alecwangcq/show-attend-and-tell/

Result:
![alt text](https://github.com/pseulki/Deep-Learning-Paper/blob/master/3_Show_Attend_Tell/data/example_bus.png)
