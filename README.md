# Gated Transformer Model for Computer Vision

## Install
```bash
$ pip install gtrxl-torch
```

## Implementation
```python
from gtrxl_torch.gtrxl_torch import GTrXL
import torch
model = GTrXL(
    d_model=512,
    nheads=4,
    transformer_layers=1
)
input = torch.randn(32,16,512)
output = model(input)
```

### Output Dimensions
   Dimension ➯ [**Batch Size**, **Patches**, **Memory Size**]
### Saving Model
```python
  model.save()
```
### Loading Model
```python
  model.load()
```

## Parameters
- `d_model`: int.  
The number of expected features in the encoder/decoder inputs
- `nheads`: int.  
The number of heads in the multiheadattention models 
- `transformer_layers`: int.  
Number of Transformer blocks.
- `hidden_dims`: int.  
Number of hidden neurons for the postion wise MLP. 
- `n_layers`: int.  
RNN (GRU) layers. 
- `chkpt_dir`: str  default `models`.  
Directory name where model is saved.
- `network_name`: str, default `network.pt`.  
Name of the model (file) you're saving.


## Resources
- *Using a transformer model on images* ➱[ Click Here](https://arxiv.org/abs/2010.11929)
- *Alterations to the transformer model (**GTrXL**)* ➱ [Click Here](https://arxiv.org/abs/1910.06764)

