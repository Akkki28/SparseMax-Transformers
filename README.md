# SparseMax-Transformers
Built a transformer from scratch that uses a SparseMax Attention mechanism instead of the standard softmax attention used in the paper [attention is all you need](https://arxiv.org/abs/1706.03762)

 # The Transformer
 ![Transformer](https://miro.medium.com/v2/resize:fit:856/1*ZCFSvkKtppgew3cc7BIaug.png)
 
The paper ‘Attention Is All You Need’ introduces a novel architecture called Transformer. As the title indicates, it uses the attention-mechanism we saw earlier. Like LSTM, Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the previously existing sequence-to-sequence models because it does not imply any Recurrent Networks (GRU, LSTM, etc.).
The Encoder is on the left and the Decoder is on the right. Both Encoder and Decoder are composed of modules that can be stacked on top of each other multiple times, which is described by Nx in the figure. We see that the modules consist mainly of Multi-Head Attention and Feed Forward layers. The inputs and outputs (target sentences) are first embedded into an n-dimensional space since we cannot use strings directly on top of which postional encodings are added as attention mechanism shares no positional information which can be crucial in these tasks.

# The Standard Attention

![attention](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfDfE_IVfwVSVk2WgYkF6Q85jiw5GyAy_rNVjNkHZ4Tg&s)
using


 
