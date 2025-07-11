#!/usr/bin/env python3
"""
Complete TensorBoard Log Analysis Script
"""

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

log_dir = '/mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/tempp/22'
event_file = os.path.join(log_dir, 'events.out.tfevents.1752216819.f65409473c38.12690.0')

ea = EventAccumulator(event_file)
ea.Reload()

print('🔍 COMPLETE TENSORBOARD LOG ANALYSIS')
print('=' * 80)

print('\n📋 BASIC INFO:')
print(f'   Log file: {event_file}')
print(f'   Available tags: {list(ea.Tags().keys())}')
print(f'   Total scalar metrics: {len(ea.Tags()["scalars"])}')

print('\n📊 ALL SCALAR METRICS:')
for tag in sorted(ea.Tags()['scalars']):
    values = ea.Scalars(tag)
    if values:
        first_val = values[0].value
        last_val = values[-1].value
        first_step = values[0].step
        last_step = values[-1].step
        total_points = len(values)
        
        print(f'   {tag}:')
        print(f'     • Points: {total_points}')
        print(f'     • Steps: {first_step} → {last_step}')
        print(f'     • Values: {first_val:.6f} → {last_val:.6f}')
        if first_val != 0:
            change_pct = ((last_val - first_val) / abs(first_val)) * 100
            print(f'     • Change: {change_pct:+.2f}%')
        print()

print('\n🎯 TRAINING PROGRESSION:')
train_loss = ea.Scalars('train_loss_step_step')
grad_norm = ea.Scalars('grad_norm')
lr_values = ea.Scalars('lr-AdamW')

print('Step | Train Loss | Grad Norm | Learning Rate')
print('-' * 50)
for i in range(0, len(train_loss), max(1, len(train_loss)//20)):
    step = train_loss[i].step
    loss = train_loss[i].value
    grad = grad_norm[i].value if i < len(grad_norm) else 0
    lr = lr_values[i].value if i < len(lr_values) else 0
    print(f'{step:4d} | {loss:10.6f} | {grad:9.3f} | {lr:.6f}')

print('\n🔄 VALIDATION METRICS:')
val_metrics = ['val_loss', 'val_recon_tf', 'val_spec_tf', 'val_cross', 'val_amp_phase']
for metric in val_metrics:
    if metric in ea.Tags()['scalars']:
        values = ea.Scalars(metric)
        if values:
            print(f'   {metric}: {values[-1].value:.6f} (step {values[-1].step})')

print('\n🎭 CORRELATION ANALYSIS:')
for tf in ['m1', 'm5', 'm15', 'm30', 'h1', 'h4']:
    tag = f'val_corr_{tf}'
    if tag in ea.Tags()['scalars']:
        values = ea.Scalars(tag)
        if values:
            all_values = [v.value for v in values]
            print(f'   {tf.upper()}: Current={values[-1].value:.6f}, Min={min(all_values):.6f}, Max={max(all_values):.6f}')

print('\n⚡ STABILITY METRICS:')
amp_overflow = ea.Scalars('amp_overflow')
grad_clipped = ea.Scalars('grad_norm_clipped')
print(f'   AMP overflows: {amp_overflow[-1].value:.0f}')
print(f'   Gradient clipping ratio: {grad_clipped[-1].value:.6f}')

print('\n📈 LOSS COMPONENT EVOLUTION:')
loss_components = ['train_recon_tf', 'train_spec_tf', 'train_cross', 'train_amp_phase']
for component in loss_components:
    if component in ea.Tags()['scalars']:
        values = ea.Scalars(component)
        if values and len(values) > 1:
            evolution = [v.value for v in values[-5:]]
            print(f'   {component}: {evolution}')

print('\n🕐 TIMING ANALYSIS:')
train_steps = [v.step for v in train_loss]
if len(train_steps) > 1:
    step_intervals = [train_steps[i+1] - train_steps[i] for i in range(len(train_steps)-1)]
    print(f'   Average steps between logs: {sum(step_intervals)/len(step_intervals):.1f}')
    print(f'   Step range: {min(train_steps)} - {max(train_steps)}')
    print(f'   Total training steps: {max(train_steps)}')

print('\n🔥 COMPLETE DATA DUMP:')
print('=' * 50)
for tag in sorted(ea.Tags()['scalars']):
    values = ea.Scalars(tag)
    if values:
        print(f'\n{tag}:')
        if len(values) <= 10:
            # Show all values for short series
            for v in values:
                print(f'  Step {v.step}: {v.value:.6f}')
        else:
            # Show first 5 and last 5 for long series
            print('  First 5:')
            for v in values[:5]:
                print(f'  Step {v.step}: {v.value:.6f}')
            print('  ...')
            print('  Last 5:')
            for v in values[-5:]:
                print(f'  Step {v.step}: {v.value:.6f}')

print('\n🔥 PERFORMANCE SUMMARY:')
print(f'   • Model: T5-large (2.1B parameters)')
print(f'   • Architecture: Model v2 async multi-scale')
print(f'   • Training steps completed: {len(train_loss)}')
print(f'   • Loss reduction: {((train_loss[-1].value - train_loss[0].value) / train_loss[0].value * 100):+.1f}%')
print(f'   • Current gradient norm: {grad_norm[-1].value:.3f}')
print(f'   • Learning rate: {lr_values[-1].value:.6f}')
print(f'   • Training stability: {"✅ Stable" if grad_norm[-1].value < 2.0 else "⚠️ Unstable"}')
print(f'   • Memory efficiency: {"✅ No AMP overflows" if amp_overflow[-1].value == 0 else "⚠️ AMP issues"}')