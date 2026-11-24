"""BandFormer: causal next-slot spectrum-occupancy model (DNN Challenge 2).

Each of the 6 sub-bands is modelled as its own stream with shared weights (the streams are stacked in the batch
dimension). Per stream: GRU -> causal self-attention over time -> MLP; then the 6 streams of each slot exchange
information through attention across sub-bands -> MLP; then a second causal self-attention over time -> MLP.
Output: one busy score per sub-band, mapped onto the 64 tokens through the fixed bit table (log-linear part), plus
a learned joint correction from an MLP over all 6 band states (captures correlations between sub-bands).

Course building blocks only (linear layers, GELU MLPs, LayerNorm, residual connections, GRU, causal multi-head
attention, optionally with RoPE, causal depthwise convolution). The only fixed tables are the token->bits unpacking,
the band rotation index and the RoPE cos/sin (when used), all built from formulas in __init__ as non-persistent buffers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NB = 6          # sub-bands
VOCAB = 64      # 2 ** NB tokens
MAX_LEN = 1024

# d: width of every stream. heads: heads of the two time attentions (head size d / heads). rope: rotary positions
# in the time attentions (False = they match on content alone; the GRU and the causal convolutions carry the order).
# joint_hidden: width of the joint output MLP. mult: MLP expansion (0 = no MLPs).
CONFIG = {'d': 40, 'heads': 1, 'rope': False, 'joint_hidden': 28, 'mult': 1}  # @@CONFIG@@


class CausalDWConv(nn.Module):
    """Depthwise 1-D convolution over time, left-padded: position t sees t-k+1 .. t. (N, L, C) -> (N, L, C)."""

    def __init__(self, c, k):
        super().__init__()
        self.k = k
        self.conv = nn.Conv1d(c, c, k, groups=c)

    def forward(self, x):
        y = self.conv(F.pad(x.transpose(1, 2), (self.k - 1, 0)))
        return y.transpose(1, 2).contiguous()  # contiguous: keeps attention on the fast GPU kernel


class TimeAttention(nn.Module):
    """Causal multi-head self-attention over time with a causal depthwise conv on q/k/v (Primer-style), so a key
    summarises the last few slots and a head can find 'the earlier slots that looked like now' and read what
    followed them. rope=True adds rotary positions (RoPE) to q and k."""

    def __init__(self, d, heads, conv_k=3, rope=True):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.qconv = CausalDWConv(3 * d, conv_k)
        self.out = nn.Linear(d, d, bias=False)
        self.rope = rope
        if rope:
            half = self.hd // 2
            inv = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float64) / half))
            ang = torch.arange(MAX_LEN, dtype=torch.float64).unsqueeze(1) * inv
            self.register_buffer("cos", ang.cos().to(torch.get_default_dtype()), persistent=False)
            self.register_buffer("sin", ang.sin().to(torch.get_default_dtype()), persistent=False)

    def _rope(self, t):
        L = t.shape[2]
        c, s = self.cos[:L].to(t.dtype), self.sin[:L].to(t.dtype)
        a, b = t[..., 0::2], t[..., 1::2]
        return torch.stack((a * c - b * s, a * s + b * c), -1).flatten(-2)

    def forward(self, x):
        N, L, _ = x.shape
        q, k, v = self.qconv(self.qkv(x)).view(N, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = self._rope(q), self._rope(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(N, L, -1))


class BandAttention(nn.Module):
    """Multi-head self-attention across the 6 sub-bands of the same slot (no time mixing, hence causal).
    Written out explicitly: the attention matrices are only 6 x 6, and the generic attention kernel is slow for
    hundreds of thousands of such tiny problems."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(self, x):  # (N, 6, d)
        N, S, _ = x.shape
        q, k, v = self.qkv(x).view(N, S, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)   # each (N, h, 6, hd)
        att = torch.softmax((q @ k.transpose(-1, -2)) * self.hd ** -0.5, dim=-1)       # (N, h, 6, 6)
        return self.out((att @ v).transpose(1, 2).reshape(N, S, -1))


class MLP(nn.Module):
    """Pre-LayerNorm GELU MLP."""

    def __init__(self, d, mult):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))



class BandFormer(nn.Module):
    def __init__(self, d=32, heads=2, rope=True, joint_hidden=64, mult=2, mix_heads=2, conv_k=3):
        super().__init__()
        self.d = d
        # fixed re-encodings of the current token (free: formula-built, non-persistent)
        tok = torch.arange(VOCAB).unsqueeze(1)
        self.register_buffer("bits", ((tok >> torch.arange(NB)) & 1).to(torch.get_default_dtype()), persistent=False)
        rot = (torch.arange(NB)[:, None] + torch.arange(NB)[None, :]) % NB  # stream b sees bands b, b+1, ... (mod 6)
        self.register_buffer("rot", rot, persistent=False)
        # input: the current 6 bits rotated so the stream's own band comes first, plus a learned band identity
        self.inp = nn.Linear(NB, d)
        self.band_emb = nn.Parameter(torch.zeros(NB, d))
        nn.init.normal_(self.band_emb, std=0.1)
        # per-band temporal stack (weights shared by the 6 streams)
        self.gru_n, self.gru = nn.LayerNorm(d), nn.GRU(d, d, batch_first=True)
        self.att1_n, self.att1 = nn.LayerNorm(d), TimeAttention(d, heads, conv_k, rope)
        self.mlp1 = MLP(d, mult) if mult else None
        # exchange between the 6 sub-bands of each slot
        self.mix_n, self.mix = nn.LayerNorm(d), BandAttention(d, mix_heads)
        self.mlp2 = MLP(d, mult) if mult else None
        # second temporal layer, now aware of the other bands (hoppers, contention)
        self.att2_n, self.att2 = nn.LayerNorm(d), TimeAttention(d, heads, conv_k, rope)
        self.mlp3 = MLP(d, mult) if mult else None
        # output: per-band busy scores + joint 64-way correction
        self.norm_f = nn.LayerNorm(d)
        self.band_head = nn.Linear(d, 1)
        self.joint1 = nn.Linear(NB * d, joint_hidden)
        self.joint2 = nn.Linear(joint_hidden, VOCAB)
        nn.init.zeros_(self.joint2.weight)
        nn.init.zeros_(self.joint2.bias)

    def forward(self, x):  # x: (B, L) int64 -> (B, L, 64) logits for the next slot
        B, L = x.shape
        d = self.d
        dt = self.inp.weight.dtype
        bits = self.bits.to(dt)[x]                                           # (B, L, 6)
        h = self.inp(bits[..., self.rot]) + self.band_emb                    # (B, L, 6 streams, d)
        h = h.permute(0, 2, 1, 3).reshape(B * NB, L, d)                      # one stream per (sequence, band)
        h = h + self.gru(self.gru_n(h))[0]
        h = h + self.att1(self.att1_n(h))
        if self.mlp1 is not None:
            h = h + self.mlp1(h)
        h = h.view(B, NB, L, d).transpose(1, 2).reshape(B * L, NB, d)        # group the 6 bands of each slot
        h = h + self.mix(self.mix_n(h))
        if self.mlp2 is not None:
            h = h + self.mlp2(h)
        h = h.view(B, L, NB, d).transpose(1, 2).reshape(B * NB, L, d)
        h = h + self.att2(self.att2_n(h))
        if self.mlp3 is not None:
            h = h + self.mlp3(h)
        h = self.norm_f(h).view(B, NB, L, d).transpose(1, 2)                 # (B, L, 6, d)
        s = self.band_head(h).squeeze(-1)                                    # (B, L, 6) per-band busy scores
        logits = s @ self.bits.to(dt).t()                                    # independent-bands part
        return logits + self.joint2(F.gelu(self.joint1(h.reshape(B, L, NB * d))))


def build_model():
    return BandFormer(**CONFIG)
