# T5転移学習 vs ランダム初期化 比較実験レポート

**実験日時**: Sun Jul  6 20:19:47 JST 2025  
**タイムスタンプ**: 20250706_193255

## 実験設定

| 実験名 | 設定ファイル | 説明 |
|--------|-------------|------|
| T5-Freeze | t5_freeze_10ep.yaml | T5を2エポック凍結後解凍 |
| T5-NoFreeze | t5_nofreeze_10ep.yaml | T5を最初から学習 |
| T5-FrozenAll | t5_frozen_all_10ep.yaml | T5を完全凍結（表現のみ利用） |
| Baseline | t5_baseline_10ep.yaml | ランダム初期化（従来手法） |

## 実験結果

### T5凍結2エポック→解凍(10ep)

```
実験: T5凍結2エポック→解凍(10ep)
設定: t5_freeze_10ep.yaml
ステータス: SUCCESS
実行時間: 830秒

最終メトリクス:
                                                                        [AEpoch 3: 100%|██████████| 100/100 [01:45<00:00,  0.94it/s, val_loss_ep=197.8837, val_loss=211.9845, val_corr_ep=-0.015, val_corr=+0.032, lr=7.02e-04]Epoch 3: 100%|██████████| 100/100 [01:45<00:00,  0.94it/s, val_loss_ep=197.8837, val_loss=211.9845, val_corr_ep=-0.015, val_corr=+0.032, lr=7.02e-04]Epoch 3: 100%|██████████| 100/100 [01:47<00:00,  0.93it/s, val_loss_ep=197.8837, val_loss=211.9845, val_corr_ep=-0.015, val_corr=+0.032, lr=7.02e-04]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=0.0093.ckpt
     val_correlation       -0.025717243552207947
  val_correlation_mean     -0.025717243552207947
        val_loss             197.8602752685547
```

### T5凍結なし(10ep)

```
実験: T5凍結なし(10ep)
設定: t5_nofreeze_10ep.yaml
ステータス: SUCCESS
実行時間: 1012秒

最終メトリクス:
                                                                        [AEpoch 5: 100%|██████████| 100/100 [01:51<00:00,  0.90it/s, val_loss_ep=3357.7207, val_loss=3357.9482, val_corr_ep=+0.028, val_corr=-0.024, lr=3.80e-04]Epoch 5: 100%|██████████| 100/100 [01:51<00:00,  0.90it/s, val_loss_ep=3357.7207, val_loss=3357.9482, val_corr_ep=+0.028, val_corr=-0.024, lr=3.80e-04]Epoch 5: 100%|██████████| 100/100 [01:52<00:00,  0.89it/s, val_loss_ep=3357.7207, val_loss=3357.9482, val_corr_ep=+0.028, val_corr=-0.024, lr=3.80e-04]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=02-val_correlation_mean=0.0482.ckpt
     val_correlation       0.030611896887421608
  val_correlation_mean     0.030611896887421608
        val_loss             3357.674560546875
```

### T5完全凍結(10ep)

```
実験: T5完全凍結(10ep)
設定: t5_frozen_all_10ep.yaml
ステータス: SUCCESS
実行時間: 593秒

最終メトリクス:
                                                                        [AEpoch 3: 100%|██████████| 100/100 [01:52<00:00,  0.89it/s, val_loss_ep=200.1127, val_loss=217.0939, val_corr_ep=-0.012, val_corr=+0.034, lr=7.02e-04]Epoch 3: 100%|██████████| 100/100 [01:52<00:00,  0.89it/s, val_loss_ep=200.1127, val_loss=217.0939, val_corr_ep=-0.012, val_corr=+0.034, lr=7.02e-04]Epoch 3: 100%|██████████| 100/100 [01:53<00:00,  0.88it/s, val_loss_ep=200.1127, val_loss=217.0939, val_corr_ep=-0.012, val_corr=+0.034, lr=7.02e-04]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=0.0094.ckpt
     val_correlation       -0.022112784907221794
  val_correlation_mean     -0.022112784907221794
        val_loss            200.09133911132812
```

### ベースライン(10ep)

```
実験: ベースライン(10ep)
設定: t5_baseline_10ep.yaml
ステータス: SUCCESS
実行時間: 377秒

最終メトリクス:
                                                                        [AEpoch 3: 100%|██████████| 100/100 [00:59<00:00,  1.69it/s, val_loss_ep=74.7349, val_loss=69.1995, val_corr_ep=-0.037, val_corr=+0.066, lr=7.02e-04]  Epoch 3: 100%|██████████| 100/100 [00:59<00:00,  1.69it/s, val_loss_ep=74.7349, val_loss=69.1995, val_corr_ep=-0.037, val_corr=+0.066, lr=7.02e-04]Epoch 3: 100%|██████████| 100/100 [00:59<00:00,  1.69it/s, val_loss_ep=74.7349, val_loss=69.1995, val_corr_ep=-0.037, val_corr=+0.066, lr=7.02e-04]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=0.0050.ckpt
     val_correlation       -0.03581126779317856
  val_correlation_mean     -0.03581126779317856
        val_loss             74.75943756103516
```

## 分析

### 完全凍結T5の効果（T5-FrozenAll）
**期待傾向**:
- **val_loss**: ランダム初期化より速く下がるが、途中から横ばい
- **val_correlation**: 0 → +0.02 程度で頭打ち
- **grad_norm**: 低い（0.5 前後）- T5部分が学習されないため

### 収束速度比較
- エポック1, 3, 5での各メトリクスの値を比較
- T5転移学習による早期収束の効果を確認
- 完全凍結での初期ブースト効果を測定

### 最終性能比較
- エポック10での最終val_correlationを比較
- T5の事前学習済み知識の純粋な効果（完全凍結）
- 微調整による追加効果（Freeze vs NoFreeze vs FrozenAll）

### 計算効率
- 実行時間の比較（完全凍結は最も高速）
- gradient norm パターンの違い
- Early Stopping による効率化効果

## 結論

[手動で分析結果を記入してください]
- T5事前学習表現の純粋効果: FrozenAll vs Baseline
- 微調整の追加価値: Freeze/NoFreeze vs FrozenAll
- 最適なT5適応戦略の特定

## TensorBoardログ

各実験のTensorBoardログは以下にあります：
- `logs/stage1/` ディレクトリ内の各バージョン

比較表示コマンド：
```bash
tensorboard --logdir logs/stage1/ --port 6006
```

## 再現方法

同じ実験を再実行する場合：
```bash
# 個別実行例
python3 scripts/train_stage1.py --config configs/t5_freeze.yaml --data_dir ../data/derived --max_epochs 10

# 自動実験再実行
bash scripts/run_t5_experiments.sh
```
