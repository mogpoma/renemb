"""Library of convenience functions/modules for Transformer encoders and GAN models.
"""

import contextlib
import matplotlib.pyplot as plt
import logging
import math
from shutil import get_terminal_size

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from contextlib import suppress
import seaborn as sns

torch.set_printoptions(linewidth=get_terminal_size()[0])
logger = logging.getLogger(__name__)

def confusion_matrix_figure(predicted, target, vocab):
    """
    Plot a confusion matrix with an heatmap and returns the figure
    cm: confusion matrix where rows are target and columns are predicted
    """
    y_pred = predicted.astype(int).tolist()
    y_true = target.astype(int).tolist()
    labels = list(sorted(set(y_true)))
    labels += list(sorted([x for x in set(y_pred) if x not in labels]))
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    cm = cm[~np.all(cm == 0, axis=1)]

    cm_labels = vocab.lookup_tokens(labels)
    with contextlib.suppress(ValueError):
        cm_labels[cm_labels.index(" ")] = "(SPC)"
    with contextlib.suppress(ValueError):
        cm_labels[cm_labels.index("\t")] = "(TAB)"
    with contextlib.suppress(ValueError):
        cm_labels[cm_labels.index("[UNK]")] = "(None)"

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, 
                cmap='binary', 
                linecolor='black',
                linewidths=.25,
                robust=True,
                xticklabels=cm_labels, 
                yticklabels=cm_labels[:len(cm[:,0])], 
                annot=True, 
                fmt='d', 
                cbar=False,
                square=True, 
                ax=ax
                )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")

    return fig


def f1_table(predicted_dict, target_dict):
    
    res = {}
    for k in predicted_dict.keys():
        p, r, f1, s = precision_recall_fscore_support(target_dict[k], predicted_dict[k], average="weighted", zero_division=0)
        res[k] = {"precision": p, "recall": r, "f1": f1, "support": s} 

    accuracy = 0
    n_samples = len(predicted_dict["delimiter"])
    for idx in range(n_samples):
        if predicted_dict["delimiter"][idx] == target_dict["delimiter"][idx] \
                and predicted_dict["quotechar"][idx] == target_dict["quotechar"][idx] \
                and predicted_dict["escapechar"][idx] == target_dict["escapechar"][idx]:
            accuracy += 1
    
    accuracy = accuracy / n_samples
    
    table = f"| Class | F1 | Precision | Recall | Support |\n"
    table += f"| --- | --- | --- | --- | --- |\n"
    for k in res.keys():
        table += f"| {k} | {res[k]['f1']:.4f} | {res[k]['precision']:.4f} | {res[k]['recall']:.4f} | {res[k]['support'] or ''} |\n"
    table += f"| Accuracy | {accuracy:.4f} | | | |\n"
    return table


def get_attn_pad_mask(seq_q, seq_k, PAD_INDEX=0):
    """This function generates a mask for the input tokens, containing 1 in the positions that are padded
    """
    len_q = seq_q.size()[-1]
    pad_attn_mask = seq_k.data.eq(PAD_INDEX)  # ... x 1 x len_k(=len_q), one is masking
    # return pad_attn_mask.unsqueeze(-2).repeat(*([1]*len(seq_q.size()[:-1])), len_q, 1)  # batch_size x len_q x len_k

def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LGCYScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        """
        :param d_k: Dimension of embedding
        """
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attn_mask):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, v)
        return context, attn


class LGCYMultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_k: int,
                 d_v,
                 n_heads: int):
        """
        :param d_model: embedding size
        :param d_k, d_v:  dimension of K(=Q), V
        :param n_heads: number of heads
        """
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k, self.d_v = d_k, d_v
        self.context_linear = nn.Linear(n_heads * d_v, d_model)
        self.context_layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask):
        # In self attention Q,K,V are the same vectror
        # q: [batch_size x len_k x d_model],
        # k: [batch_size x len_k x d_model],
        # v: [batch_size x len_v x d_model]
        residual = q
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # Here we have a linear layer which produces (..., n_heads*d_k) vectors.
        # We want to split them into (..., n_heads, n_layers)
        q_s = self.W_Q(q)
        q_s = q_s.view(*q_s.size()[:-2], -1, self.n_heads, self.d_k).transpose(-2,-3)

        k_s = self.W_K(k)
        k_s = k_s.view(*k_s.size()[:-2], -1, self.n_heads, self.d_k).transpose(-2,-3)

        v_s = self.W_V(v)
        v_s = v_s.view(*v_s.size()[:-2], -1, self.n_heads, self.d_v).transpose(-2, -3)  # v_s: [batch_size x n_heads x len_k x d_v]

        # attn_mask : [batch_size x n_heads x len_q x len_k]
        attn_mask = attn_mask.unsqueeze(-3).repeat(*([1]*len(attn_mask.size()[:-2])), self.n_heads, 1, 1)

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        attn = torch.matmul(q_s, k_s.transpose(-1,-2))
        attn = attn / np.sqrt(self.d_k)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        attn.masked_fill_(attn_mask, 1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(attn)

        context = torch.matmul(attn, v_s)
        context = context.transpose(-3, -2).contiguous().view(*context.size()[:-3], -1, self.n_heads * self.d_v)
        # context: [batch_size x len_q x n_heads * d_v]
        output = self.context_linear(context)
        output = self.context_layer_norm(output + residual)
        return output, attn  # output: [batch_size x len_q x d_model]

class LGCYEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_k: int, d_v: int,
                 n_heads: int, d_ff: int):
        super(EncoderLayer, self).__init__()
        self.encoder_self_attention = MultiHeadAttention(d_model=d_model, d_k=d_k, d_v=d_v, n_heads=n_heads)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        The encoder contains self-attention layers.
        In a self-attention layer all the keys, values and queries come from the same place, in this case, the output of the previous layer in the
        encoder. Each position in the encoder can attend to all positions in the previous layer of the
        encoder.
        """

        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.encoder_self_attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        enc_outputs = self.fc1(enc_outputs)
        enc_outputs = gelu(enc_outputs)
        enc_outputs = self.fc2(enc_outputs)
        # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn
