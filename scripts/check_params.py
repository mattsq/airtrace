import torch
from airtrace.models.informer import InformerModel

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_config(d_model, nhead, e_layers, d_layers, ff_dim):
    # Input/Output dims from dataset (13 in, 11 out)
    model = InformerModel(
        input_dim=13,
        output_dim=11,
        d_model=d_model,
        nhead=nhead,
        e_layers=e_layers,
        d_layers=d_layers,
        ff_dim=ff_dim,
        factor=5,
        dropout=0.1,
        pred_len=1,
        distill=True
    )
    params = count_params(model)
    print(f"Config: d_model={d_model}, nhead={nhead}, layers={e_layers}/{d_layers}, ff={ff_dim}")
    print(f"Parameters: {params:,}")
    return params

print("--- Parameter Search Round 2 ---")
# Attempt 3 modified
check_config(256, 8, 3, 2, 1024)

# Slightly deeper
check_config(256, 8, 4, 2, 1024)