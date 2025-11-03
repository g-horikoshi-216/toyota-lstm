# Improved LSTM v2: Simplified Model Implementation

## 変更内容

### 背景
前回の改良版LSTM（v1）は以下の問題を抱えていました：
- **過度に複雑**：BiLSTM、LayerNormalization、Huber損失など、小規模データ（900日）に対して過剰な構造
- **低いR²スコア**：0.14（標準LSTMの0.30より悪化）
- **過学習の兆候**：複雑さが裏目に出た可能性

### 改良版 v2 の戦略

#### 1. **シンプルな構造**
```python
model_reg_v2 = Sequential([
    LSTM(96, return_sequences=True),    # BiLSTM → 単方向LSTM
    Dropout(0.20),                       # 30% → 20%
    BatchNormalization(),                # LayerNorm → BatchNorm
    LSTM(64, return_sequences=False),
    Dropout(0.15),                       # 25% → 15%
    Dense(32, activation='relu'),        # swish → relu
    Dense(1, activation='linear')
])
```

**主な変更点：**
- `Bidirectional` → 単方向 `LSTM`（パラメータ数削減）
- `LayerNormalization` → `BatchNormalization`（学習安定化）
- Dropout率の削減（0.30/0.25 → 0.20/0.15）
- 活性化関数を`swish` → `relu`に単純化

#### 2. **MSE損失への回帰**
```python
model_reg_v2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Huber → MSE
    metrics=['mae']
)
```

- `Huber(delta=1.0)` → `mse`
- 複雑な学習率スケジューラ（CosineDecayRestarts）を削除
- シンプルな固定学習率（0.001）

#### 3. **標準的なコールバック**
```python
callbacks_v2 = [
    EarlyStopping(patience=20),
    ReduceLROnPlateau(factor=0.5, patience=8),
    ModelCheckpoint('toyota_lstm_v2_best.keras')
]
```

- `AdamW`の複雑な設定を削除
- 標準的な`Adam`オプティマイザを使用

### 追加された機能

#### 1. モデル定義・学習セル
- v2モデルの構築と学習
- 学習曲線の可視化

#### 2. 評価セル
- テストセットでの予測
- RMSE、R²スコアの計算
- 予測値 vs 実測値のグラフ

#### 3. 分類・バックテストセル
- Buy/Sell信号の生成
- 混同行列の可視化
- 累積リターンのグラフ

#### 4. 全モデル比較セル
- Baseline、Standard LSTM、v1、v2の統一比較
- 回帰性能、分類性能、バックテスト結果を一覧表示

## 期待される効果

### ✅ 改善が期待される点
1. **R²スコアの向上**：シンプル化により過学習を抑制
2. **安定した学習**：BatchNormalizationによる勾配の安定化
3. **適切な信号割合**：過度な楽観予測の抑制

### ⚠️ トレードオフ
1. **表現力の低下**：BiLSTMの双方向性を失う
2. **外れ値への感度**：Huber損失の外れ値耐性を失う

## 実行方法

### セルの実行順序
1. データ読み込み〜特徴量生成（既存セル）
2. 時系列分割・ウィンドウデータセット作成（既存セル）
3. **新規：改良版LSTM v2 モデル定義・学習**
4. **新規：改良版LSTM v2 評価**
5. **新規：改良版LSTM v2 分類・バックテスト**
6. **新規：全モデル比較**

### 注意事項
- 標準LSTMのセルを先に実行してください（`close_test_tail`などの変数が必要）
- 比較セルは全モデルの結果が揃ってから実行してください

## 評価指標

### 目標値
- **R² > 0.30**：標準LSTMと同等以上
- **Buy信号割合 40-60%**：実際の分布（37.9%）に近づける
- **Accuracy > 50%**：ランダムより良い性能
- **Sharpe Ratio > 0**：リスク調整後でプラスのリターン

## 次のステップ

### v2で改善された場合
1. ハイパーパラメータのさらなる調整
2. テスト期間の延長（3ヶ月→6ヶ月）
3. アンサンブル学習の検討

### v2でも改善しない場合
1. 特徴量の見直し（外部要因の追加）
2. ウィンドウサイズの調整（30日→20日、50日）
3. 異なるアーキテクチャの検討（GRU、Transformer）

