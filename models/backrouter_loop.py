"""
Backward-depth-router looped GPT (a variant of base_loop.py).

Idea: in a recurrent-depth (looped) transformer, "how many recurrences a token needs" is a
discrete per-token decision, structurally like MoE expert choice. The forward halting decision
is bounded (you can't know if another step helps until you see the loss); but the BACKWARD pass
has the target, so the *marginal value* of each step  delta_d = loss_{d-1} - loss_d  is exactly
computable. We supervise a forward halting head with that target-aware marginal value (the
"backward router for depth"), and deep-supervise every depth so the shared block is anytime-usable.

Why this may clear the wall that blocks MoE value-routing: "how much compute a token needs"
correlates with token difficulty / predictive entropy / syntactic role -- which ARE predictable
from x -- unlike "which expert is best" (surprise-bound). So the forward policy the backward
router supervises has a much higher ceiling.

Self-contained (single file), mirrors base_loop.py for clarity.
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
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.intermediate_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.intermediate_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_1 = nn.RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = nn.RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))
        return x


class SharedBlock(nn.Module):
    def __init__(self, depth, config):
        super().__init__()
        self.blocks = nn.ModuleList([Block(config) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class GPTConfig:
    model_type: str = 'backrouter_loop'
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 3
    n_head: int = 32
    n_embd: int = 2048
    dropout: float = 0.0
    bias: bool = False
    intermediate_dim: int = 5120
    max_model_loops: int = 8     # K: max recurrences
    halt_coeff: float = 0.1      # weight of the halting BCE (the backward router)
    halt_tau: float = 0.0        # marginal-value threshold: step worth it iff delta_d > tau
    deep_supervision: bool = True  # supervise every depth (anytime-usable block)
    halt_ponder_coeff: float = 0.0  # ponder cost: penalize expected continue-mass (compute)
    halt_head_type: str = 'linear'  # 'linear' (feedforward) or 'gru' (state-tracking over depth)
    halt_gru_dim: int = 256      # hidden size of the GRU halting head (if 'gru')


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None and config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=SharedBlock(config.n_layer, config),
            norm_f=nn.RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Halting head (the forward depth policy): from the state BEFORE a step, predict whether
        # that step is worth running. Supervised by the backward-computed marginal value.
        # 'gru' tracks the value/state trajectory across recurrences (recurrent backward router).
        if config.halt_head_type == 'gru':
            self.halt_gru = nn.GRUCell(config.n_embd, config.halt_gru_dim)
            self.halt_proj = nn.Linear(config.halt_gru_dim, 1, bias=True)
        else:
            self.halt_head = nn.Linear(config.n_embd, 1, bias=True)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _readout_loss(self, x, targets):
        """Per-token CE of an early exit at the current depth. Returns ([B,T] loss, logits)."""
        logits = self.lm_head(self.transformer.norm_f(x))            # [B,T,V]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction='none').view(targets.shape)  # [B,T]
        return loss, logits

    def forward(self, idx, targets=None, steps=None):
        device = idx.device
        b, t = idx.size()
        K = steps if steps is not None else self.config.max_model_loops
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))

        if targets is None:
            # Inference: run K steps (adaptive per-token halting is exercised in generate()).
            for _ in range(K):
                x = self.transformer.h(x)
            logits = self.lm_head(self.transformer.norm_f(x[:, [-1], :]))
            return logits, None, x

        # --- training: per-depth early-exit loss + backward-router halting supervision ---
        use_gru = self.config.halt_head_type == 'gru'
        gru_state = None
        loss_prev, _ = self._readout_loss(x, targets)               # depth-0 readout [B,T]
        lm_terms = [loss_prev.mean()] if self.config.deep_supervision else []
        halt_terms, ponder_terms = [], []
        logits = None
        for _d in range(1, K + 1):
            if use_gru:  # recurrent backward router: track the value/state trajectory over depth
                gru_state = self.halt_gru(x.reshape(b * t, -1), gru_state)
                halt_logit = self.halt_proj(gru_state).view(b, t)   # [B,T] is step _d worth it?
            else:
                halt_logit = self.halt_head(x).squeeze(-1)          # [B,T]
            x = self.transformer.h(x)                               # apply the shared (recurrent) block
            loss_d, logits = self._readout_loss(x, targets)         # [B,T], [B,T,V]
            # Backward router: marginal value of this step (target-aware), supervises halting.
            delta = (loss_prev - loss_d).detach()                  # >0 => step helped
            halt_target = (delta > self.config.halt_tau).float()
            halt_terms.append(F.binary_cross_entropy_with_logits(halt_logit, halt_target))
            ponder_terms.append(torch.sigmoid(halt_logit).mean())  # expected continue-mass (compute)
            lm_terms.append(loss_d.mean())
            loss_prev = loss_d

        lm_loss = torch.stack(lm_terms).mean() if self.config.deep_supervision else lm_terms[-1]
        halt_loss = torch.stack(halt_terms).mean()
        ponder_loss = torch.stack(ponder_terms).mean()
        loss = (lm_loss + self.config.halt_coeff * halt_loss
                + self.config.halt_ponder_coeff * ponder_loss)
        return logits, loss, x

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        return flops_per_iter * (1.0 / dt) / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, max_steps=None,
                 halt_thresh=0.5):
        """Adaptive-depth generation: at each recurrence, halt the per-token loop early when the
        halting head says the next step is not worth it (sigmoid < halt_thresh). Demonstrates the
        learned x-based depth policy at inference."""
        K = max_steps if max_steps is not None else self.config.max_model_loops
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            pos = torch.arange(0, idx_cond.size(1), dtype=torch.long, device=idx.device)
            x = self.transformer.wte(idx_cond) + self.transformer.wpe(pos)
            use_gru = self.config.halt_head_type == 'gru'
            gru_state = None
            for _d in range(K):
                if use_gru:
                    gru_state = self.halt_gru(x[:, -1, :], gru_state)
                    cont = torch.sigmoid(self.halt_proj(gru_state)).mean().item()
                else:
                    cont = torch.sigmoid(self.halt_head(x[:, -1:, :])).mean().item()  # batch-mean halting
                if cont < halt_thresh:
                    break
                x = self.transformer.h(x)
            logits = self.lm_head(self.transformer.norm_f(x[:, -1, :])) / temperature  # [B,V]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
