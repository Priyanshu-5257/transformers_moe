This repo is forked from orginal transformers repo
### Modifications :
- It has modified bert model with mixture experts layers
- Instead of using mixture of experts over transformer head, we have used it in MLP layer within the each head
- There are total 5 MLP experts in which 2 are participating at a time. 


```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model_ = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

config = model_.config
from transformers import BertModel, BertTokenizer
model = BertModel(config=config,moe = True)
```
### Example 
Kaggle Notebook Link[https://www.kaggle.com/code/hbpkillerx/moe-bert-1]
