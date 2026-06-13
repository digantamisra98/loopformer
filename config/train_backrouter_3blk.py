# Backward-depth-router looped GPT — matched to train_base_loop_3blk.py (same size/data) so the
# comparison is clean: does value-supervised adaptive depth beat fixed-depth looping at matched
# average compute? Backward router supervises a halting head with the per-step marginal value.

batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

max_iters = 50000
lr_decay_iters = 50000

out_dir = "runs/backrouter_3blk"
dataset = "fineweb_edu"

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 2e-1

### model cfg
model_type = 'backrouter_loop'
max_model_loops = 8
n_layer = 3
n_head = 32
n_embd = 2048
