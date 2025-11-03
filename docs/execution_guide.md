# Toyota LSTM Notebook 実行ガイド

## 📋 セルの実行順序

### Phase 1: データ準備
1. **ライブラリのインポート**
2. **データ読み込み関数の定義** (`load_csv_data`)
3. **CSVからデータ読み込み** (`data/TM_1980-01-01_2025-06-27.csv`)
4. **データ読み込み結果の可視化**

### Phase 2: 特徴量エンジニアリング
5. **テクニカル指標の計算** (RSI, MACD, Bollinger Bands, MA)
6. **テクニカル指標の詳細可視化**

### Phase 3: データ分割
7. **時系列分割** (Train/Val/Test)
   - Train: 2020/09〜2024/03
   - Val: 2024/04〜2025/03
   - Test: 2025/04〜2025/09
8. **データ分割の可視化**
9. **ウィンドウデータセット作成** (30日ウィンドウ)

### Phase 4: ベースラインモデル
10. **線形回帰ベースライン** (回帰タスク)

### Phase 5: 標準LSTM
11. **LSTM モデル定義・学習** (回帰)
12. **学習曲線の可視化**
13. **テストセット評価**

### Phase 6: 改良版LSTM v1 (複雑版)
14. **Improved LSTM v1 モデル定義・学習** (BiLSTM, Huber損失)
15. **学習曲線の可視化**
16. **モデル比較可視化** (Baseline vs Standard vs v1)

### Phase 7: 改良版LSTM v2 (簡略版) ⭐️ NEW
17. **Markdown: v2の戦略説明**
18. **🆕 Improved LSTM v2 モデル定義・学習**
   - 単方向LSTM
   - MSE損失
   - BatchNormalization
19. **🆕 Improved LSTM v2 評価**
   - RMSE, R² 計算
   - 学習曲線の可視化
   - 予測値 vs 実測値のプロット
20. **🆕 Improved LSTM v2 分類・バックテスト**
   - Buy/Sell信号生成
   - 混同行列
   - 累積リターン
21. **🆕 全モデル比較**
   - Baseline / Standard / v1 / v2 の統一比較
   - 回帰・分類・バックテストの全指標

### Phase 8: 分類タスク (標準LSTM)
22. **分類：Buy/Sell判定**
23. **分類性能の可視化**
24. **予測の信頼度分析**
25. **予測結果の詳細可視化**

### Phase 9: バックテスト
26. **シンプル・バックテスト**
27. **バックテストの詳細可視化**

### Phase 10: 保存と診断
28. **モデル・スケーラーの保存**
29. **モデル全体の精度サマリー可視化**
30. **問題診断レポート**

---

## ⚠️ 重要な注意事項

### 依存関係
以下のセルは**順番に実行**する必要があります（並列実行不可）：

1. **v2分類セル（Phase 7-20）は Phase 8-22 の後に実行**
   - `close_test_tail` 変数が必要
   - 標準LSTMの分類セルを先に実行してください

2. **全モデル比較セル（Phase 7-21）は全モデル実行後**
   - すべてのモデルの結果変数が必要
   - 最後に実行することを推奨

### 推奨実行パターン

#### パターンA: すべて順番に実行
```
Phase 1 → Phase 2 → ... → Phase 10
```
最も確実な方法。初回実行時に推奨。

#### パターンB: v2だけ実行
既に標準LSTM・v1を実行済みの場合：
```
Phase 1-6 (スキップ可)
↓
Phase 7: v2のみ実行
↓
Phase 8-22: 分類セル実行（close_test_tail生成のため）
↓
Phase 7-20, 21: v2分類・比較実行
```

#### パターンC: モデル比較のみ更新
すべてのモデルを実行済みの場合：
```
Phase 7-21: 全モデル比較セルのみ再実行
```

---

## 📊 期待される結果

### 現在の結果（2025年4-6月テスト期間）

| モデル | RMSE | R² | Accuracy | Buy信号% |
|:--|:--:|:--:|:--:|:--:|
| Baseline | 5.48 | 0.26 | - | - |
| Standard LSTM | 5.18 | 0.30 | 37.9% | 72.4% |
| Improved v1 | 5.75 | 0.14 | 48.3% | 75.9% |
| **Improved v2** | **未実行** | **未実行** | **未実行** | **未実行** |

### v2の目標
- **R² > 0.30**: 標準LSTMと同等以上
- **Buy信号 40-60%**: 実際の分布（37.9%）に近づける
- **Accuracy > 50%**: ランダムより良い
- **Sharpe Ratio > 0**: リスク調整後でプラス

---

## 🐛 トラブルシューティング

### `NameError: name 'close_test_tail' is not defined`
**原因**: v2の分類セルを Phase 8-22 より前に実行した

**解決策**:
```python
# Phase 8-22 の分類セルを先に実行
# その後 Phase 7-20 の v2分類セルを実行
```

### `NameError: name 'metrics_te' is not defined`
**原因**: 比較セルを他のモデル実行前に実行した

**解決策**: すべてのモデルセルを先に実行してから比較セルを実行

### `KeyError` や `NameError` が頻発
**原因**: セルの実行順序が不適切

**解決策**: カーネルを再起動して Phase 1 から順番に実行

---

## 💡 ヒント

### 学習時間の目安
- 標準LSTM: 約5-10分（Epoch 100）
- Improved v1: 約10-15分（複雑な構造）
- Improved v2: 約5-10分（シンプル化）

### GPU推奨
TensorFlowがGPUを認識している場合、学習が大幅に高速化されます。
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### メモリ不足の場合
`BATCH_SIZE` を 32 → 64 に増やすか、`WINDOW_SIZE` を 30 → 20 に減らす

---

## 📚 参考資料

- [v2モデルの詳細](v2_model_changes.md)
- [プロジェクトレポート](Toyota_LSTM_BuySell_Report.md)
- Jupyter Notebook: `toyota_lstm_buy_sell.ipynb`
