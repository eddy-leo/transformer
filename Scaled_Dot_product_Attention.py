import torch
import torch.nn.functional as F
import math


def Scaled_Dot_Product_Attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    k = query.size(-1)
    scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))

    if mask is not None:
        scores = scores.masked_fill(mask==0, -math.inf)

    weights = F.softmax(scores, dim=-1)

    return torch.matmul(weights, value)