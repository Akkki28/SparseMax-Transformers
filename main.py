from Transformer import Transformer
import torch
input_dim = 32     
d_model = 64       
num_layers = 3     
nhead = 8          
num_classes = 5       
batch_size = 16
seq_len = 20
dummy_input = torch.randn(batch_size, seq_len, input_dim)   
mask = None
   
model = Transformer(input_dim,d_model,num_layers,nhead,dim_feedforward=128,dropout=0.1,num_classes=num_classes)
output, attentions = model(dummy_input, mask)
print("Model output shape:", output.shape)