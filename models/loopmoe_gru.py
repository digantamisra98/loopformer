"""Looped MoE with a GRU forward router (idea A: routing consistency across recurrences).

In a looped block with experts, the router fires at every recurrence on the evolving state, so a
token can hit a different expert each loop -> the shared experts get conflicting updates across
recurrence-phases (gradient conflict) and the expert path is incoherent. Fix: a GRU carries the
routing state ACROSS recurrences (more natural than RMoE's cross-layer GRU, since a looped block
reuses the same weights), giving a coherent expert path. Dense-compute prototype (all experts
evaluated, gated) for correctness; swap to a sparse dispatcher to scale.
"""
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd, self.dropout = config.n_head, config.n_embd, config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.moe_inter_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.moe_inter_dim, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class MoEMLP(nn.Module):
    """Top-k MoE whose router is a GRU carrying routing state across recurrences."""
    def __init__(self, config):
        super().__init__()
        self.n_experts, self.top_k = config.num_experts, config.moe_top_k
        self.router_type = config.router_type
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        if self.router_type == 'gru':  # carries routing state across recurrences (consistency)
            self.router_gru = nn.GRUCell(config.n_embd, config.router_gru_dim)
            self.router_proj = nn.Linear(config.router_gru_dim, config.num_experts)
        else:  # plain per-recurrence router (control: can flip-flop across loops)
            self.router_lin = nn.Linear(config.n_embd, config.num_experts)

    def forward(self, x, state):
        B, T, C = x.shape
        xf = x.reshape(B * T, C)
        if self.router_type == 'gru':
            state = self.router_gru(xf, state)                  # [BT, gru] routing state over depth
            logits = self.router_proj(state)                    # [BT, N]
        else:
            logits = self.router_lin(xf)                        # [BT, N], stateless
        topv, topi = logits.topk(self.top_k, dim=-1)            # [BT, k]
        # softmax upcasts to fp32 under autocast; cast back so scatter dtype matches logits
        gate_full = torch.zeros_like(logits).scatter_(
            1, topi, torch.softmax(topv, dim=-1).to(logits.dtype))  # [BT, N]
        out = torch.zeros_like(xf)
        for j, expert in enumerate(self.experts):               # dense-compute prototype
            out = out + gate_full[:, j:j + 1] * expert(xf)
        # switch load-balance aux: N * sum_j (frac_j * meanprob_j)
        probs = torch.softmax(logits, dim=-1)
        frac = (gate_full > 0).float().mean(0)
        aux = self.n_experts * (frac * probs.mean(0)).sum()
        return out.view(B, T, C), state, aux


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = nn.RMSNorm(config.n_embd)
        self.moe = MoEMLP(config)

    def forward(self, x, state):
        x = x + self.attn(self.norm_1(x))
        m, state, aux = self.moe(self.norm_2(x), state)
        x = x + m
        return x, state, aux


class SharedBlock(nn.Module):
    def __init__(self, depth, config):
        super().__init__()
        self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(depth)])

    def forward(self, x, states):
        aux = 0.0
        for i, block in enumerate(self.blocks):
            x, states[i], a = block(x, states[i])
            aux = aux + a
        return x, states, aux


@dataclass
class GPTConfig:
    model_type: str = 'loopmoe_gru'
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 3
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = False
    intermediate_dim: int = 1536   # kept for interface compat (unused by experts)
    max_model_loops: int = 6
    num_experts: int = 8
    moe_top_k: int = 2
    moe_inter_dim: int = 1024
    router_gru_dim: int = 256
    moe_aux_coeff: float = 0.01
    router_type: str = 'gru'   # 'gru' (state across recurrences) or 'linear' (stateless control)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=SharedBlock(config.n_layer, config),
            norm_f=nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, steps=None):
        b, t = idx.size()
        K = steps if steps is not None else self.config.max_model_loops
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        states = [None] * self.config.n_layer  # per-block GRU routing state, persists across loops
        aux_total = 0.0
        for _d in range(K):
            x, states, aux = self.transformer.h(x, states)
            aux_total = aux_total + aux
        x = self.transformer.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = loss + self.config.moe_aux_coeff * (aux_total / (K * self.config.n_layer))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss, x

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        pd = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for n, p in pd.items() if p.dim() >= 2]
        nodecay = [p for n, p in pd.items() if p.dim() < 2]
        groups = [{'params': decay, 'weight_decay': weight_decay},
                  {'params': nodecay, 'weight_decay': 0.0}]
        fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device_type == 'cuda'
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **(dict(fused=True) if fused else {}))

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops = (6 * N + 12 * L * H * Q * T) * T * fwdbwd_per_iter
        return flops * (1.0 / dt) / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, max_steps=None):
        K = max_steps if max_steps is not None else self.config.max_model_loops
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond, None, steps=K)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
