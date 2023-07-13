from transformers import BertModel
import torch

class KREL(torch.nn.Module):
    def __init__(self, n_classes=116, dim_k=768, dim_v=768):
        super(KREL, self).__init__()
        self.model_name = 'KREL'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.fcc_tar = torch.nn.Linear(768, n_classes)
        self.fcc_rel = torch.nn.Linear(768, n_classes)
        self.fcc_sub = torch.nn.Linear(768, n_classes)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(3)])
        self.n_classes = n_classes

    def encode(self, target_ids, rel_ids, sub_ids):
        att_tar = (target_ids>0)
        _, tar = self.bert_model(input_ids=target_ids, attention_mask=att_tar, return_dict=False)
        att_rel = (rel_ids>0)
        _, rel = self.bert_model(input_ids=rel_ids, attention_mask=att_rel, return_dict=False)
        att_sub = (sub_ids>0)
        _, sub = self.bert_model(input_ids=sub_ids, attention_mask=att_sub, return_dict=False)

        return tar, rel, sub
    
    def forward(self,tar_ids,rel_ids, sub_ids):
        tar, rel, sub = self.encode(tar_ids, rel_ids, sub_ids)
        tar_out = self.dropout(tar)
        rel_out = self.dropout(rel)
        sub_out = self.dropout(sub)
        out_tar = self.fcc_tar(tar_out)
        out_rel = self.fcc_rel(rel_out)
        out_sub = self.fcc_sub(sub_out)
        res = self.weights[0]*out_tar+self.weights[1]*out_rel+self.weights[2]*out_sub
        return res
