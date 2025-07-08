# T5転移学習 vs ランダム初期化 比較実験レポート

**実験日時**: Sun Jul  6 17:18:41 JST 2025  
**タイムスタンプ**: 20250706_163259

## 実験設定

| 実験名 | 設定ファイル | 説明 |
|--------|-------------|------|
| Baseline | t5_baseline.yaml | ランダム初期化（従来手法） |
| T5-Freeze | t5_freeze.yaml | T5を3エポック凍結後解凍 |
| T5-NoFreeze | t5_nofreeze.yaml | T5を最初から学習 |

## 実験結果

### T5凍結3エポック

```
実験: T5凍結3エポック
設定: t5_freeze.yaml
実行時間: 592秒

最終メトリクス:
                                                                        [AEpoch 3: 100%|██████████| 100/100 [01:49<00:00,  0.92it/s, val_loss_ep=1489.1488, val_loss=1503.8018, val_corr_ep=-0.098, val_corr=-0.113, lr=4.85e-04, grad_norm=1.72e+00, amp=0]Epoch 3: 100%|██████████| 100/100 [01:49<00:00,  0.92it/s, val_loss_ep=1489.1488, val_loss=1503.8018, val_corr_ep=-0.098, val_corr=-0.113, lr=4.85e-04, grad_norm=1.72e+00, amp=0]Epoch 3: 100%|██████████| 100/100 [01:51<00:00,  0.90it/s, val_loss_ep=1489.1488, val_loss=1503.8018, val_corr_ep=-0.098, val_corr=-0.113, lr=4.85e-04, grad_norm=1.72e+00, amp=0]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=-0.0550.ckpt
     val_correlation       -0.09907375276088715
  val_correlation_mean     -0.09907375276088715
        val_loss            1489.0972900390625
```

### T5凍結なし

```
実験: T5凍結なし
設定: t5_nofreeze.yaml
実行時間: 608秒

最終メトリクス:
Epoch 3: 100%|██████████| 100/100 [01:58<00:00,  0.84it/s, val_loss_ep=3390.6084, val_loss=3400.2527, val_corr_ep=-0.074, val_corr=-0.248, lr=2.97e-04, grad_norm=1.00e+03, amp=0]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=00-val_correlation_mean=-0.0229.ckpt
     val_correlation        -0.0906745120882988
  val_correlation_mean      -0.0906745120882988
        val_loss             3390.609130859375
```

### ベースライン(ランダム初期化)

```
実験: ベースライン(ランダム初期化)
設定: t5_baseline.yaml
実行時間: 1542秒

最終メトリクス:
                                                                        [AEpoch 9: 100%|██████████| 100/100 [01:59<00:00,  0.84it/s, val_loss_ep=92.4833, val_loss=79.7520, val_corr_ep=-0.080, val_corr=-0.127, lr=5.44e-04, grad_norm=4.18e+04, amp=0]Epoch 9: 100%|██████████| 100/100 [01:59<00:00,  0.84it/s, val_loss_ep=92.4833, val_loss=79.7520, val_corr_ep=-0.080, val_corr=-0.127, lr=5.44e-04, grad_norm=4.18e+04, amp=0]Epoch 9: 100%|██████████| 100/100 [01:59<00:00,  0.84it/s, val_loss_ep=92.4833, val_loss=79.7520, val_corr_ep=-0.080, val_corr=-0.127, lr=5.44e-04, grad_norm=4.18e+04, amp=0]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=06-val_correlation_mean=-0.0604.ckpt
     val_correlation       -0.09060180187225342
  val_correlation_mean     -0.09060180187225342
        val_loss             92.57075500488281
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
