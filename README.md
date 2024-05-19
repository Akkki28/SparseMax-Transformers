# SparseMax-Transformers
Built a transformer from scratch that uses a SparseMax Attention mechanism instead of the standard softmax attention used in the paper [attention is all you need](https://arxiv.org/abs/1706.03762)

 # The Transformer
 ![Transformer](https://miro.medium.com/v2/resize:fit:856/1*ZCFSvkKtppgew3cc7BIaug.png)
 
The paper ‘Attention Is All You Need’ introduces a novel architecture called Transformer. As the title indicates, it uses the attention-mechanism we saw earlier. Like LSTM, Transformer is an architecture for transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the previously existing sequence-to-sequence models because it does not imply any Recurrent Networks (GRU, LSTM, etc.).
The Encoder is on the left and the Decoder is on the right. Both Encoder and Decoder are composed of modules that can be stacked on top of each other multiple times, which is described by Nx in the figure. We see that the modules consist mainly of Multi-Head Attention and Feed Forward layers. The inputs and outputs (target sentences) are first embedded into an n-dimensional space since we cannot use strings directly on top of which postional encodings are added as attention mechanism shares no positional information which can be crucial in these tasks.

# The Standard Attention

![attention](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfDfE_IVfwVSVk2WgYkF6Q85jiw5GyAy_rNVjNkHZ4Tg&s)

Attention mechanisms compute attention scores between words in the input sequence, determining the importance of each word for a specific word. These scores are used to create weighted representations, which are then combined to form the output.Attention mechanisms are crucial because they allow Transformers to capture long-range dependencies, process sequences in parallel, and adapt to various domains. They are foundational for natural language processing, computer vision, and more tasks.

# SparseMax
Sparsemax is a type of activation/output function similar to the traditional softmax, but able to output sparse probabilities.

![sparseMax](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCwhRmjTVO4PtsJ0zn4RWZAMmAZ5cy-haAaelP9NquTg&s)

the ability to generate sparse probability distributions. This means it can assign high probabilities to a select few elements while driving the rest to zero is what differs SparsMax models from softmax as softmax can never output 0 probablities. This can be beneficial when you want the model to focus on a limited number of relevant features or input aspects and not attend on others.



![SParseMax2](https://github.com/Akkki28/SparseMax-Transformers/assets/120105455/ff9078d6-d7ce-40e6-8336-a8dca8a3689c)
 ![softmax2](https://github.com/Akkki28/SparseMax-Transformers/assets/120105455/07a97a72-a4e4-4f45-8782-dc55c767a32a)

