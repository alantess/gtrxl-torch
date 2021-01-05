# Gated Transformer Model for Computer Vision

## Install
```bash
$ pip install gtrxl-torch
```

## Implementation
```python
import torch
from gtrxl_torch.gtrxl_torch import GTrXL
transformer = GTrXL(1024,4,1,9,3)
input = T.randn((32,16,1024),device=device)
output = transformer.forward(input)
```
## Parameters
- `d_model`: int.  
The number of expected features in the encoder/decoder inputs
- `nheads`: int.  
The number of heads in the multiheadattention models 
- `n_outputs`: int.  
Number of output neurons.
- `transformer_layers`: int.  
Number of Transformer blocks.
- `fc2_dims`: int.  
Number of Output Neurons for the first FC layer & input for the output Layer
- `n_layers`: int.  
RNN (GRU) layers. 
- `lr`: float, default `0.00025`.  
Learning Rate. 
- `chkpt_dir`: str  default `model`.  
Directory name where model is saved.
- `network_name`: str, default `network`.  
Name of the model (file) you're saving.


## Resources
- *Using a transformer model on images* ➱[ Click Here](https://arxiv.org/abs/2010.11929)
- *Alterations to the transformer model (**GTrXL**)* ➱ [Click Here](https://arxiv.org/abs/1910.06764)

