"""Read-only adapter to the same-round E0 checkpoint. Never creates an optimizer."""
from pathlib import Path
import numpy as np
import torch

from debias.io import model_modules, sha256, read_json
from debias.training import build_model
from .scoring import score_rec


def load_frozen(state, source, device='cpu'):
    source = Path(source)
    marker = read_json(source/'training_complete.json')
    if marker['best_sha256'] != sha256(source/'best.pt'):
        raise ValueError('E0 checkpoint checksum mismatch')
    original = model_modules()
    bundle = original.load_semantic_feature_bundle(state.graph, None, device=torch.device(device))
    ck = torch.load(source/'best.pt', map_location=device, weights_only=False)
    if ck['config']['ablation'] != 'feature_only' or ck['config'].get('include_test_triples', False):
        raise ValueError('Expected the original feature-only E0, training graph only')
    model = build_model(original, bundle, ck['config']).to(device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.eval()
    return model, bundle


def export_logits(model, bundle, state, uids, batch_size=32):
    rows = []
    with torch.no_grad():
        for start in range(0, len(uids), batch_size):
            rows.append(score_rec(model, bundle, uids[start:start+batch_size], len(state.q)).cpu().double().numpy())
    return np.concatenate(rows)
