# T5転移学習 vs ランダム初期化 比較実験レポート

**実験日時**: Sun Jul  6 21:38:23 JST 2025  
**タイムスタンプ**: 20250706_203942

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
実行時間: 1218秒

最終メトリクス:
Epoch 9: 100%|██████████| 100/100 [01:47<00:00,  0.93it/s, val_loss_ep=161.4298, val_loss=159.4267, val_corr_ep=-0.027, val_corr=+0.331, lr=4.00e-06]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=09-val_correlation_mean=-0.0273.ckpt
     val_correlation       -0.025471877306699753
  val_correlation_mean     -0.025471877306699753
        val_loss             161.4302520751953
```

### T5凍結なし(10ep)

```
実験: T5凍結なし(10ep)
設定: t5_nofreeze_10ep.yaml
ステータス: SUCCESS
実行時間: 1061秒

最終メトリクス:
                                                                        [AEpoch 7: 100%|██████████| 100/100 [01:50<00:00,  0.90it/s, val_loss_ep=2576.1560, val_loss=2578.5591, val_corr_ep=-0.016, val_corr=+0.046, lr=1.09e-04]Epoch 7: 100%|██████████| 100/100 [01:50<00:00,  0.90it/s, val_loss_ep=2576.1560, val_loss=2578.5591, val_corr_ep=-0.016, val_corr=+0.046, lr=1.09e-04]Epoch 7: 100%|██████████| 100/100 [01:51<00:00,  0.89it/s, val_loss_ep=2576.1560, val_loss=2578.5591, val_corr_ep=-0.016, val_corr=+0.046, lr=1.09e-04]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=04-val_correlation_mean=-0.0104.ckpt
     val_correlation       -0.02092820033431053
  val_correlation_mean     -0.02092820033431053
        val_loss              2576.169921875
```

### T5完全凍結(10ep)

```
実験: T5完全凍結(10ep)
設定: t5_frozen_all_10ep.yaml
ステータス: SUCCESS
実行時間: 1018秒

最終メトリクス:
                                                                        [AEpoch 7: 100%|██████████| 100/100 [01:47<00:00,  0.93it/s, val_loss_ep=163.2950, val_loss=160.6226, val_corr_ep=-0.023, val_corr=+0.048, lr=1.09e-04]Epoch 7: 100%|██████████| 100/100 [01:47<00:00,  0.93it/s, val_loss_ep=163.2950, val_loss=160.6226, val_corr_ep=-0.023, val_corr=+0.048, lr=1.09e-04]Epoch 7: 100%|██████████| 100/100 [01:51<00:00,  0.90it/s, val_loss_ep=163.2950, val_loss=160.6226, val_corr_ep=-0.023, val_corr=+0.048, lr=1.09e-04]Using 16bit Automatic Mixed Precision (AMP)
   最良チェックポイント: /mnt/c/Users/taki/Desktop/my-projects/GoriProject/stage1/checkpoints/stage1-epoch=04-val_correlation_mean=-0.0185.ckpt
     val_correlation       -0.03739108145236969
  val_correlation_mean     -0.03739108145236969
        val_loss            163.29859924316406
```

### ベースライン(10ep)

```
実験: ベースライン(10ep)
設定: t5_baseline_10ep.yaml
ステータス: FAILED
実行時間: 223秒

実験が失敗したため、メトリクスは利用できません
エラー詳細は training.log を確認してください
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
