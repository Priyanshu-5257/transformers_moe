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
In this example notebook[https://www.kaggle.com/code/hbpkillerx/moe-bert-1], I used the sentence-transformers/all-MiniLM-L6-v2 as the base model configuration and then created the MoE-BERT model. To speed up the training process, I copied the weights from the original model’s layers and froze them during training. This way, only the newly introduced MoE layers are trained during fine-tuning, which significantly reduces the computational cost and training time.

#### Traning Results 
[[https://api.wandb.ai/links/hbpkillerx/misvgem5]]

### Changes made in 'src/transformers/models/bert/modeling_bert.py'


```python
class TopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        #layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
    
    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class SparseMoE(nn.Module):
    def __init__(self,config, expert: nn.Module, gate: nn.Module):
        super(SparseMoE, self).__init__()
        self.router = gate
        self.experts = nn.ModuleList([expert for _ in range(5)])
        self.top_k = 2
        self.intermediate_size = config.intermediate_size

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros([x.shape[0], x.shape[1], self.intermediate_size]).to(x.device)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))
        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output

class BertLayer(nn.Module):
    def __init__(self, config,moe=False):
    ...
      if self.moe :
              self.intermediate = SparseMoE(config,BertIntermediate(config), TopkRouter(config.hidden_size,5, 2))
          if not self.moe:
              self.intermediate = BertIntermediate(config)
    ...
```
