# Tensor shape ledger

This file is populated from `artifacts/qwen05b-model-audit.json` after the Hub revision is resolved.

| Tensor/module | Logical shape | Coordinate domain | Transformation |
|---|---|---|---|
| `embed_tokens.weight` | `[vocab, hidden]` | token x residual | vocabulary rows |
| `q_proj.weight` | `[q_heads * head_dim, hidden]` | Q x residual | head/block + coordinate |
| `k_proj.weight` | `[kv_heads * head_dim, hidden]` | K x residual | KV group + coordinate |
| `v_proj.weight` | `[kv_heads * head_dim, hidden]` | V x residual | KV group + coordinate |
| `o_proj.weight` | `[hidden, q_heads * head_dim]` | residual x attention | inverse attention map |
| `gate_proj.weight` | `[intermediate, hidden]` | FFN x residual | shared gate/up permutation |
| `up_proj.weight` | `[intermediate, hidden]` | FFN x residual | shared gate/up permutation |
| `down_proj.weight` | `[hidden, intermediate]` | residual x FFN | inverse FFN permutation |
| `lm_head.weight` | `[vocab, hidden]` | token x residual | vocabulary rows |

