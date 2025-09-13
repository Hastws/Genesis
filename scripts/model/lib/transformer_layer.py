import torch
import torch.nn as nn
import torch.nn.functional as F

debug = False


class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1, bias_value=-1e9):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.biasval = bias_value

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, self.biasval)
            del mask

        attn = self.dropout(F.softmax(attn, dim=-1))
        r = torch.matmul(attn, v)

        return r, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

    def forward(self, q, k, v, mask=None):
        d_k = self.d_k
        d_v = self.d_v
        n_head = self.n_head

        size_b = q.size(0)
        len_q = q.size(1)
        len_k = k.size(1)
        len_v = v.size(1)

        assert len_k == len_v, "MultiHeadAttention requires lk == lv"

        q = self.w_qs(q).view(size_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(size_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        context, attn = self.attention(q, k, v, mask=mask)
        context = context.transpose(1, 2).contiguous().view(size_b, len_q, -1)

        out = self.fc(context)

        return out, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        return x


def test():
    B, L, d_model, H, Dk, Dv = 2, 8, 64, 4, 16, 16
    mha = MultiHeadAttention(H, d_model, Dk, Dv)
    x = torch.randn(B, L, d_model, requires_grad=True)
    out, attn = mha(x, x, x, mask=torch.ones(B, L, L, dtype=torch.bool))
    loss = out.pow(2).mean()
    loss.backward()
    print(out.shape, attn.shape) if debug else None


if __name__ == "__main__":
    test()
