# T5転移学習 vs ランダム初期化 比較実験レポート

**実験日時**: Sun Jul  6 12:38:24 JST 2025  
**タイムスタンプ**: 20250706_121211

## 実験設定

| 実験名 | 設定ファイル | 説明 |
|--------|-------------|------|
| Baseline | t5_baseline.yaml | ランダム初期化（従来手法） |
| T5-Freeze | t5_freeze.yaml | T5を3エポック凍結後解凍 |
| T5-NoFreeze | t5_nofreeze.yaml | T5を最初から学習 |

## 実験結果

### ベースライン(ランダム初期化)

```
実験: ベースライン(ランダム初期化)
設定: t5_baseline.yaml
実行時間: 471秒

最終メトリクス:
Epoch 4: 100%|██████████| 100/100 [00:52<00:00,  1.91it/s, val_loss_ep=204.4490, val_loss=209.8951, val_corr_ep=-0.062, val_corr=-0.250, lr=6.01e-06, grad_norm=8.62e+03, amp=0]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=03-val_correlation_mean=-0.0440.ckpt
     val_correlation       -0.021859774366021156
  val_correlation_mean     -0.021859774366021156
        val_loss             204.4607696533203
```

### T5凍結3エポック

```
実験: T5凍結3エポック
設定: t5_freeze.yaml
実行時間: 547秒

最終メトリクス:
                                                                        [AEpoch 3: 100%|██████████| 100/100 [01:45<00:00,  0.95it/s, val_loss_ep=273.5052, val_loss=277.9319, val_corr_ep=-0.071, val_corr=-0.230, lr=6.03e-05, grad_norm=7.88e+04, amp=0]Epoch 3: 100%|██████████| 100/100 [01:45<00:00,  0.95it/s, val_loss_ep=273.5052, val_loss=277.9319, val_corr_ep=-0.071, val_corr=-0.230, lr=6.03e-05, grad_norm=7.88e+04, amp=0]Epoch 3: 100%|██████████| 100/100 [01:47<00:00,  0.93it/s, val_loss_ep=273.5052, val_loss=277.9319, val_corr_ep=-0.071, val_corr=-0.230, lr=6.03e-05, grad_norm=7.88e+04, amp=0]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=-0.0625.ckpt
     val_correlation       -0.08378248661756516
  val_correlation_mean     -0.08378248661756516
        val_loss            273.47357177734375
```

### T5凍結なし

```
実験: T5凍結なし
設定: t5_nofreeze.yaml
実行時間: 555秒

最終メトリクス:
                                                                        [AEpoch 3: 100%|██████████| 100/100 [01:39<00:00,  1.01it/s, val_loss_ep=3319.3474, val_loss=3333.3623, val_corr_ep=-0.017, val_corr=-0.052, lr=3.94e-05, grad_norm=2.61e+04, amp=0]Epoch 3: 100%|██████████| 100/100 [01:39<00:00,  1.01it/s, val_loss_ep=3319.3474, val_loss=3333.3623, val_corr_ep=-0.017, val_corr=-0.052, lr=3.94e-05, grad_norm=2.61e+04, amp=0]Epoch 3: 100%|██████████| 100/100 [01:41<00:00,  0.98it/s, val_loss_ep=3319.3474, val_loss=3333.3623, val_corr_ep=-0.017, val_corr=-0.052, lr=3.94e-05, grad_norm=2.61e+04, amp=0]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=-0.0102.ckpt
     val_correlation      -0.0010532577289268374
  val_correlation_mean    -0.0010532577289268374
        val_loss             3319.34619140625
```

## 分析

### 収束速度
- エポック1, 3, 5での各メトリクスの値を比較
- T5転移学習による早期収束の効果を確認

### 最終性能
- エポック10での最終val_correlationを比較
- T5の事前学習済み知識の効果を評価

### 計算コスト
- 実行時間の比較
- メモリ使用量の違い（ログから推定）

## 結論

[手動で分析結果を記入してください]

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
