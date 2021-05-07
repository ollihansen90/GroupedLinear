# grouped_linear
Linear layer that uses groups (similar to conv-layers) in PyTorch.

The idea is surprisingly simple: Just unsqueeze the tensor in the last dimension such that each entry of the input tensor is a channel. This can be seen as having column vectors instead of line vectors. Next apply a Conv1d with kernel size 1 and the appropriate input- and output-dimension. In the end unsqueeze to get the desired output. 

The grouped linear layer can be used like a linear layer. In fact: groups=1 is the "normal" nn.Linear.
