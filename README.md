# 🚗 Toyota Stock LSTM Prediction System

LSTMによるトヨタ自動車株価の翌日終値予測と売買判断システム

**DL4E Final Project** | 堀越 源之介 | 2025年11月

---

## 📊 プロジェクト概要

このプロジェクトは、**LSTM（Long Short-Term Memory）**を用いてトヨタ自動車（7203.T）の株価を予測し、翌日の売買判断（Buy/Sell）を行うシステムです。

### 目的
1. 翌日の終値をLSTMで予測（回帰タスク）
2. 予測結果に基づく売買判断（分類タスク）
3. 予測精度・取引戦略の有効性を検証

---

## 🎯 主な成果

### Improved LSTM v2による性能向上

| 指標 | Baseline | Standard LSTM | Improved v1 | **Improved v2** |
|:--|:--:|:--:|:--:|:--:|
| **R² Score** | 0.26 | 0.30 | 0.14 | **0.48 ✅** |
| **RMSE** | 5.48 | 5.18 | 5.75 | **4.47 ✅** |
| **Accuracy** | - | 37.9% | 48.3% | **51.7% ✅** |
| **Buy信号割合** | - | 72.4% | 75.9% | **31.0% ✅** |

**Key Achievements:**
- 🎉 **R²が60%向上**: 0.30 → 0.48（標準LSTMから）
- 🎉 **Buy信号割合が適正化**: 72.4% → 31.0%（実際は37.9%）
- 🎉 **シンプル化による汎化性能向上**: v1の過学習を解決

### 重要な発見: 「複雑なモデル ≠ 良い性能」

このプロジェクトで最も重要な技術的知見は、**モデルの複雑性とデータ量のバランス**です。

- ❌ **v1（複雑版）**: BiLSTM、LayerNorm、Huber損失 → R² = 0.14（過学習）
- ✅ **v2（簡略版）**: 単方向LSTM、BatchNorm、MSE損失 → R² = 0.48（汎化）

小規模データセット（900日）では、適度なモデル複雑性が性能向上のカギとなります。

---

## 📁 プロジェクト構成

```
toyota-lstm/
├── toyota_lstm_buy_sell.ipynb    # メインノートブック
├── data/
│   └── TM_1980-01-01_2025-06-27.csv  # トヨタ株価データ
├── docs/
│   ├── Toyota_LSTM_BuySell_Report.md    # 最終レポート
│   ├── v2_results_summary.md             # v2結果サマリー
│   ├── v2_model_changes.md               # v2詳細説明
│   └── execution_guide.md                # 実行ガイド
└── README.md                              # このファイル
```

---

## 🚀 クイックスタート

### 1. 環境構築
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
```

### 2. ノートブックを開く
```bash
jupyter notebook toyota_lstm_buy_sell.ipynb
```

### 3. セルを順番に実行
詳細は [docs/execution_guide.md](docs/execution_guide.md) を参照

---

## 📈 モデル構成

### Improved LSTM v2（推奨）
```python
model_reg_v2 = Sequential([
    LSTM(96, return_sequences=True, input_shape=(30, num_features)),
    Dropout(0.20),
    BatchNormalization(),
    LSTM(64, return_sequences=False),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# MSE損失、Adam最適化
model_reg_v2.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

**主な特徴:**
- 単方向LSTM（BiLSTMから簡略化）
- BatchNormalization（学習安定化）
- MSE損失（シンプルで安定）
- Dropout 20%, 15%（過学習抑制）

---

## 📊 結果の可視化

ノートブックには以下の可視化が含まれています：

1. **データ読み込み結果**: 株価推移、出来高
2. **テクニカル指標**: RSI、MACD、ボリンジャーバンド
3. **データ分割**: Train/Val/Test期間の可視化
4. **学習曲線**: Loss推移、Val Lossとの比較
5. **予測vs実測**: テストセットでの予測精度
6. **混同行列**: 分類タスクの評価
7. **累積リターン**: バックテスト結果
8. **全モデル比較**: 4モデルの統一比較

---

## 📚 ドキュメント

### レポート
- **[Toyota_LSTM_BuySell_Report.md](docs/Toyota_LSTM_BuySell_Report.md)**
  - プロジェクト全体の最終レポート
  - 背景、手法、結果、考察

### 技術ドキュメント
- **[v2_results_summary.md](docs/v2_results_summary.md)**
  - v2モデルの実行結果サマリー
  - 性能比較、技術的インサイト

- **[v2_model_changes.md](docs/v2_model_changes.md)**
  - v2モデルの詳細説明
  - アーキテクチャの変更点、期待される効果

- **[execution_guide.md](docs/execution_guide.md)**
  - ノートブックの実行ガイド
  - セルの実行順序、トラブルシューティング

---

## 🔬 技術スタック

- **Python 3.x**
- **TensorFlow / Keras**: LSTMモデル構築
- **Pandas**: データ処理
- **NumPy**: 数値計算
- **Matplotlib / Seaborn**: 可視化
- **Scikit-learn**: 評価指標、スケーリング

---

## 📖 主な技術要素

### データ前処理
- StandardScaler（特徴量 + ターゲット変数）
- 時系列ウィンドウ（30日）
- Train/Val/Test分割（時系列を保持）

### 特徴量エンジニアリング
- テクニカル指標: RSI、MACD、ボリンジャーバンド
- 移動平均: MA7、MA30
- 出来高変化率
- 価格変化率

### モデル評価
- 回帰: RMSE、R²スコア
- 分類: Accuracy、Precision、Recall、F1
- バックテスト: 累積リターン、シャープレシオ、最大ドローダウン

---

## 🎓 学びのポイント

### 1. モデル複雑性とデータ量のバランス
- 小規模データ（900日）には適度なモデルが最適
- BiLSTMより単方向LSTMが効果的だった

### 2. スケーリングの重要性
- ターゲット変数の適切なスケーリング + 逆変換が必須
- 別々のスケーラー（scaler_X、scaler_y）を使用

### 3. 時系列データの難しさ
- 株価は非定常性が強い
- 外部要因（ニュース、為替）の影響が大きい
- テスト期間の市場環境に大きく左右される

### 4. 評価の多面性
- 単一指標だけでは不十分
- Buy信号割合、診断レポートなど総合的に評価

---

## 🚧 今後の改善予定

### 短期的
1. テスト期間の延長（29日 → 6ヶ月以上）
2. 閾値調整（Buy判定の最適化）
3. ウィンドウサイズの実験（20日、50日）

### 中長期的
1. 外部要因の追加（為替、日経平均、ニュース）
2. Attention機構の導入
3. リスク管理機能の統合

---

## 📜 ライセンス

このプロジェクトは教育目的のものです。

---

## 👤 著者

**堀越 源之介**
- DL4E Final Project
- 2025年11月

---

## 🙏 謝辞

- データソース: Yahoo! Finance
- DL4USコースの知見を活用
- TensorFlow/Kerasコミュニティに感謝

---

## 📞 お問い合わせ

質問や提案がある場合は、Issueを作成してください。

---

**注意**: このシステムは研究・教育目的のものであり、実際の投資判断には使用しないでください。株式投資は自己責任で行ってください。
