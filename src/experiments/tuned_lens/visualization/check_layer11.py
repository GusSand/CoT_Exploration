#!/usr/bin/env python3
import json

with open('layer_probe_visualization_FIXED.json') as f:
    data = json.load(f)

ex = data['results'][0]['continuous_thoughts'][0]
l11 = ex['layers'][11]

print('Layer 11 Original Top Predictions:')
for i in range(5):
    token = l11['original']['topk_tokens'][i]
    prob = l11['original']['topk_probs'][i]
    print(f'  {repr(token)}: {prob:.4f}')

print(f'\nTop prob: {l11["original"]["top1_prob"]:.4f}')
print(f'Top token: {repr(l11["original"]["top1_token"])}')

# Also check Layer 10 for comparison
l10 = ex['layers'][10]
print('\nLayer 10 Original Top Predictions:')
for i in range(5):
    token = l10['original']['topk_tokens'][i]
    prob = l10['original']['topk_probs'][i]
    print(f'  {repr(token)}: {prob:.4f}')
