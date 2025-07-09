#!/usr/bin/env python3
from tensorboard.backend.event_processing import event_accumulator
import numpy as np

print('📊 詳細ログ解析 - 全メトリクス')
print('=' * 60)

ea = event_accumulator.EventAccumulator('tempp/events.out.tfevents.1752068954.04becf7deb90.1600.0')
ea.Reload()

scalar_tags = ea.Tags()['scalars']

# 損失成分の詳細分析
loss_components = [tag for tag in scalar_tags if 'loss' in tag.lower()]
print(f'📉 損失成分分析 ({len(loss_components)}個):')
for tag in sorted(loss_components):
    events = ea.Scalars(tag)
    if events:
        values = [e.value for e in events]
        if values[0] != 0:
            improvement = (values[0]-values[-1])/values[0]*100
        else:
            improvement = 0
        print(f'   {tag}: 初期={values[0]:.4f} → 最終={values[-1]:.4f} (改善率={improvement:.1f}%)')

# 相関メトリクス
corr_tags = [tag for tag in scalar_tags if 'corr' in tag.lower()]
print(f'\n📈 相関メトリクス ({len(corr_tags)}個):')
for tag in sorted(corr_tags):
    events = ea.Scalars(tag)
    if events:
        values = [e.value for e in events]
        change = values[-1]-values[0]
        print(f'   {tag}: 初期={values[0]:.6f} → 最終={values[-1]:.6f} (変化={change:+.6f})')

# 一貫性メトリクス
consistency_tags = [tag for tag in scalar_tags if 'consistency' in tag.lower()]
print(f'\n🔗 一貫性メトリクス ({len(consistency_tags)}個):')
for tag in sorted(consistency_tags):
    events = ea.Scalars(tag)
    if events:
        values = [e.value for e in events]
        print(f'   {tag}: 初期={values[0]:.6f} → 最終={values[-1]:.6f}')

# スペクトラムメトリクス
spec_tags = [tag for tag in scalar_tags if 'spec' in tag.lower()]
print(f'\n🌈 スペクトラムメトリクス ({len(spec_tags)}個):')
for tag in sorted(spec_tags):
    events = ea.Scalars(tag)
    if events:
        values = [e.value for e in events]
        print(f'   {tag}: 初期={values[0]:.4f} → 最終={values[-1]:.4f}')

# 再構成メトリクス（TF別）
recon_tags = [tag for tag in scalar_tags if 'recon' in tag.lower()]
print(f'\n🔄 再構成メトリクス ({len(recon_tags)}個):')
for tag in sorted(recon_tags):
    events = ea.Scalars(tag)
    if events:
        values = [e.value for e in events]
        print(f'   {tag}: 初期={values[0]:.4f} → 最終={values[-1]:.4f}')

# 学習率の詳細（主要層のみ）
key_lr_tags = [tag for tag in scalar_tags if 'lr-AdamW' in tag and ('block_0' in tag or 'block_23' in tag or 'head_and_adapter' in tag)]
print(f'\n📊 主要層学習率:')
for tag in sorted(key_lr_tags):
    events = ea.Scalars(tag)
    if events:
        values = [e.value for e in events]
        layer_name = tag.split('/')[-1] if '/' in tag else tag.split('-')[-1]
        ratio = values[-1]/values[0] if values[0] != 0 else 0
        print(f'   {layer_name}: 初期={values[0]:.2e} → 最終={values[-1]:.2e} (倍率={ratio:.0f}x)')

# 全メトリクス一覧
print(f'\n📋 全メトリクス名一覧 ({len(scalar_tags)}個):')
for tag in sorted(scalar_tags):
    events = ea.Scalars(tag)
    print(f'   {tag} ({len(events)}ポイント)')