This repo is forked from orginsl transformers repo
### Modifications :
- It has modified bert model with mixture experts layers
- Instead of using mixture of experts over transformer head, we have used it in MLP layer within the each head
- There are total 5 MLP experts in which 2 are participating at a time. 
