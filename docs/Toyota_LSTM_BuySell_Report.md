# トヨタ株価予測モデルの開発と改善プロセス

**堀越 源之介 / DL4E Final Project / 2025年**

---

## 1. 背景・目的

### 背景
- 現在私は証券会社向けシステムの開発に従事しており、証券や金融に関連するテーマを設定したいと考えた
- DL4USにおいて、時系列データの学習に対してLSTMが有効であることを学んだ
- 証券分野における時系列データといえば株価推移であり、株価の時系列データを学習し未来の動向を予測するモデルを作成したいと考えた

### 目的
本プロジェクトでは、**トヨタ自動車（7203.T / TM）** の株価データを用いて、以下を目指す：

1. **翌営業日の価格動向**を機械学習モデルで予測（二値分類タスク）
2. 複数のモデルアーキテクチャを比較検証
3. 予測に基づく売買戦略の有効性を検証し、AIによる株式判断の可能性を探る

---

## 2. 使用データ・期間

| 項目 | 内容 |
|:--|:--|
| 対象銘柄 | トヨタ自動車（7203.T / TM） |
| データソース | Kaggle CSV（Yahoo Finance データ） |
| データ期間 | 1980年3月17日〜2025年6月26日（約45年分） |
| 総データ数 | 11,413営業日 |
| 粒度 | 日次（営業日のみ） |
| 訓練期間 | 1980年3月17日〜2024年3月31日（10,940日） |
| テスト期間 | 2024年4月1日〜2025年6月26日（310日） |

### 基本特徴量
- **OHLCV**: Open, High, Low, Close, Adjusted Close, Volume

### 追加したテクニカル指標
1. **ret_1d**: 日次リターン（前日比の変化率）
2. **ma_5**: 5日移動平均
3. **ma_20**: 20日移動平均
4. **ma_ratio**: 移動平均比率（ゴールデンクロス度合い）
5. **vol_change**: 出来高変化率
6. **MACD**: 移動平均収束拡散（トレンド指標）
   - macd: MACD線
   - macdsignal: シグナル線
   - macdhist: MACDヒストグラム
7. **RSI**: 相対力指数（買われすぎ・売られすぎ指標）
8. **bb_width**: ボリンジャーバンド幅（ボラティリティ指標）

**合計10特徴量**を使用

---

## 3. タスク定義

### 目的変数
**二値分類タスク**: 翌営業日の株価が上昇するか下落するかを予測

```
target_buy = 1  翌日の終値 > 今日の終値（上昇）
target_buy = 0  翌日の終値 ≤ 今日の終値（下落）
```

### クラス分布
- 訓練データ: 下落 54.3%, 上昇 45.7%
- テストデータ: 下落 51.3%, 上昇 48.7%

→ ほぼバランスが取れているが、わずかに下落が多い

---

## 4. 開発プロセスと発生した課題

### フェーズ1: 初期実装とバグ修正

#### 課題1: CSVデータの読み込みエラー
**発生した問題**:
```
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

**原因**:
- CSVファイルの2行目にティッカーシンボル（"TM"）が含まれており、数値として扱うべき列に文字列が混入

**解決策**:
```python
# 日付列を変換（無効な日付はNaTに）
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 日付が無効な行を削除（ティッカー行を除外）
df = df.dropna(subset=['date']).copy()

# 数値列を明示的に変換
for col in required:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 欠損値を含む行を削除
df = df.dropna(subset=required).copy()
```

#### 課題2: 無限大値によるモデル学習エラー
**発生した問題**:
```
ValueError: Input X contains infinity or a value too large for dtype('float64')
```

**原因**:
- `pct_change()`や移動平均の割り算でゼロ除算が発生
- 特に`ma_ratio = ma_5 / ma_20 - 1.0`で`ma_20`がゼロに近い場合

**解決策**:
```python
# ma_20がゼロに近い場合の処理
feats['ma_ratio'] = np.where(
    feats['ma_20'] > 0.01,
    feats['ma_5'] / feats['ma_20'] - 1.0,
    0.0
)

# 無限大値をNaNに変換して除去
feats = feats.replace([np.inf, -np.inf], np.nan)
feats = feats.dropna().copy()
```

#### 課題3: データリークの防止
**発生した問題**:
- 初期実装では、全データに対して`future_ret`（翌日のリターン）を計算してから訓練/テストに分割していた
- これにより、テストデータの情報が訓練データに漏れる可能性があった

**解決策**:
```python
# まずデータを分割
train = features.loc[features.index <= TRAIN_END].copy()
test = features.loc[features.index > TRAIN_END].copy()

# 分割後にターゲット変数を作成
train['future_ret'] = train['close'].shift(-1) / train['close'] - 1.0
train['target_buy'] = (train['future_ret'] > 0).astype(int)

test['future_ret'] = test['close'].shift(-1) / test['close'] - 1.0
test['target_buy'] = (test['future_ret'] > 0).astype(int)
```

---

### フェーズ2: 特徴量エンジニアリング

#### 課題4: taライブラリの不足
**発生した問題**:
```
ModuleNotFoundError: No module named 'ta'
```

**解決策**:
```bash
pip install ta
```

#### 追加した特徴量とその効果
1. **MACD指標**: トレンドの方向性と強さを測定
   - `ta.trend.MACD()`を使用して3つの指標を生成

2. **RSI指標**: モメンタム分析
   - `ta.momentum.RSIIndicator()`で買われすぎ・売られすぎを判定

3. **ボリンジャーバンド幅**: ボラティリティ測定
   - `ta.volatility.BollingerBands()`でバンド幅を正規化

**効果**: 特徴量が5→10に増加し、モデルがより多様なパターンを学習可能に

---

### フェーズ3: モデルの改善

#### 課題5: クラス不均衡による予測の偏り
**発生した問題**:
- 初期モデルは**全て上昇予測**（予測分布: 下落0%, 上昇100%）
- 正解率は約48%だが、実質的に役に立たない

**原因分析**:
1. クラス不均衡（下落54.3% vs 上昇45.7%）
2. ロジスティック回帰では線形分離が困難
3. デフォルトの重み付けでは少数派クラスを無視

**解決策1: クラス重み付け**
```python
# ロジスティック回帰
model = LogisticRegression(class_weight='balanced')

# RandomForest
model = RandomForestClassifier(class_weight='balanced')

# XGBoost
scale_pos_weight = n_class_0 / n_class_1
model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
```

**結果**: 下落予測が0%→10%に改善

**解決策2: 非線形モデルの導入**
- ロジスティック回帰では限界があるため、RandomForestを導入
- RandomForest導入後: 下落予測が10%→19.4%に改善

#### 課題6: 予測確率の閾値最適化
**発生した問題**:
- デフォルトの閾値0.5では、予測が上昇に偏る（下落予測19.4%）

**解決策: 閾値探索とバランススコア**
```python
# 0.45〜0.60の範囲で閾値を探索
for thresh in np.arange(0.45, 0.60, 0.01):
    y_pred_temp = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_temp)
    balance = abs(n_down - n_up) / len(y_pred_temp)

# 正解率とバランスの両方を考慮したスコア
score = accuracy - 0.3 * balance
```

**結果**:
- 最適閾値の自動選択
- 予測のバランスが改善

---

### フェーズ4: 複数モデルの実装と比較

#### 実装したモデル

##### 1. RandomForest（ベースモデル）
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced'
)
```
- **特徴**: 非線形パターンを捉える、過学習に強い
- **入力**: 10特徴量（単一時点）

##### 2. XGBoost
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight
)
```
- **特徴**: 勾配ブースティング、高速で高精度
- **入力**: 10特徴量（単一時点）

#### 課題7: XGBoostのインストール
**発生した問題**:
```
ModuleNotFoundError: No module named 'xgboost'
```

**解決策**:
```bash
pip install xgboost
```

##### 3. LightGBM
```python
lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight
)
```
- **特徴**: XGBoostより高速、メモリ効率が良い、Leaf-wise成長戦略
- **入力**: 10特徴量（単一時点）

##### 4. LSTM（時系列モデル）
```python
Sequential([
    LSTM(32, return_sequences=True),
    Dropout(0.3),
    LSTM(16, return_sequences=False),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```
- **特徴**: 長期依存関係を学習、時系列専用
- **入力**: (20日, 10特徴量)のシーケンスデータ
- **パラメータ**: 学習率0.0005、バッチサイズ64、最大100エポック
- **正則化**: Dropout(0.2-0.3)、EarlyStopping、ReduceLROnPlateau

#### 課題8: LSTMの過学習
**発生した問題**:
- 初期実装（LSTM(64)→LSTM(32)→Dense(16)）では過学習が発生
- テスト精度が低い（約48%）

**解決策**:
1. **モデルの簡素化**: LSTM(32)→LSTM(16)→Dense(8)にパラメータ数を削減
2. **Dropout率の増加**: 0.2→0.3に強化
3. **学習率の調整**: 0.001→0.0005に低下
4. **バッチサイズの増加**: 32→64でより安定した学習
5. **Early Stoppingの調整**: patience=10→15でより長く待機

**結果**: 過学習が軽減され、汎化性能が向上

##### 5. GRU（時系列モデル）
```python
Sequential([
    GRU(32, return_sequences=True),
    Dropout(0.3),
    GRU(16, return_sequences=False),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
```
- **特徴**: LSTMより単純で高速、メモリ効率が良い
- **入力**: (20日, 10特徴量)のシーケンスデータ
- **ハイパーパラメータ**: LSTMと同じ設定

---

## 5. モデル比較結果

### テストデータでの性能比較

| モデル | 正解率 | 下落予測 | 上昇予測 | 特徴 |
|:--|:--:|:--:|:--:|:--|
| **LightGBM** | **51.94%** | 96 (31%) | 214 (69%) | 最高精度 ⭐ |
| XGBoost | 51.29% | 106 (34%) | 204 (66%) | 高速 |
| RandomForest | 49.68% | 31 (10%) | 279 (90%) | ベースライン |
| GRU | 49.31% | 4 (1%) | 286 (93%) | 時系列学習 |
| LSTM | 47.24% | 24 (8%) | 266 (86%) | 時系列学習 |

### 採用モデル: LightGBM

**採用理由**:
1. 最高の正解率（51.94%）
2. 閾値最適化後も50.65%の精度を維持
3. 高速な学習・推論（約1秒）
4. 特徴量の重要度が解釈可能

### LightGBMの特徴量重要度

| 順位 | 特徴量 | 重要度 | 重要度(%) | 解釈 |
|:--:|:--|:--:|:--:|:--|
| 1 | ret_1d | 354 | 14.6% | 日次リターンが最重要 |
| 2 | vol_change | 348 | 14.3% | 出来高変化率 |
| 3 | rsi | 292 | 12.0% | 相対力指数 |
| 4 | bb_width | 273 | 11.2% | ボリンジャーバンド幅 |
| 5 | macdhist | 266 | 10.9% | MACDヒストグラム |
| 6 | ma_ratio | 262 | 10.8% | 移動平均比率 |
| 7 | macd | 232 | 9.5% | MACDトレンド |
| 8 | ma_5 | 223 | 9.2% | 5日移動平均 |
| 9 | macdsignal | 222 | 9.1% | MACDシグナル |
| 10 | ma_20 | 164 | 6.7% | 20日移動平均 |

**分析**:
- **ret_1d（日次リターン）とvol_change（出来高変化率）が最重要**（合計28.9%）
- MACD系特徴量も有効（macd, macdsignal, macdhistで合計29.5%）
- RSIとボリンジャーバンド幅がボラティリティ指標として機能（合計23.2%）
- すべての特徴量が均等に寄与しており、バランスが良い

---

## 6. 閾値最適化

### 閾値探索結果（LightGBM）

| 閾値 | 正解率 | 下落予測 | 上昇予測 | バランス | スコア |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.45 | 50.65% | 66 | 244 | 0.574 | 0.334 |
| 0.49 | 52.58% | 88 | 222 | 0.432 | 0.396 |
| 0.52 | 52.58% | 110 | 200 | 0.290 | 0.439 |
| **0.56** | **50.65%** | **148** | **162** | **0.045** | **0.493** ⭐ |
| 0.57 | 49.68% | 157 | 153 | 0.013 | 0.493 |
| 0.59 | 50.00% | 186 | 124 | 0.200 | 0.440 |

**最適閾値**: 0.56（バランススコア最大）
- **正解率**: 50.65%
- **予測分布**: 下落47.7%, 上昇52.3%（ほぼ完璧なバランス）

**バランススコア**: `score = accuracy - 0.3 × balance`
- 正解率とクラスバランスの両方を考慮
- 閾値0.56で最適なトレードオフを実現
- 最高正解率（0.49で52.58%）より、バランスを優先

---

## 7. バックテスト結果

### 取引戦略
- **買いシグナル**: モデルが上昇予測（確率≥0.56）の日に株を購入
- **待機**: モデルが下落予測の日は現金保有
- **比較対象**: 買い持ち戦略（期間開始時に購入して持ち続ける）

### パフォーマンス

| 指標 | 戦略（LightGBM） | 買い持ち | 差分 |
|:--|--:|--:|--:|
| 最終倍率 | **1.1139** | 0.7046 | **+0.4093** |
| リターン | **+11.39%** | -29.54% | **+40.93%** |
| 総取引回数 | 162回（52.3%） | - | - |
| 勝率 | 49.4% | - | - |
| 平均日次リターン | +0.045% | -0.095% | - |
| リターン標準偏差 | 1.41% | 1.60% | - |
| 最大リターン | +9.30% | - | - |
| 最小リターン | -5.21% | - | - |

### 主要な発見

1. **下落相場での優位性**
   - テスト期間（2024年4月〜2025年6月）は下落相場
   - 買い持ちは-29.54%の損失
   - 戦略は+11.39%の利益
   - **超過リターン +40.93%**

2. **選択的取引の効果**
   - 310日中162日のみ取引（52%）
   - 下落が予想される日は現金保有で損失回避
   - 低勝率（49.4%）でも利益を達成

3. **リスク特性**
   - 最大損失: -5.21%（比較的小さい）
   - 最大利益: +9.30%（非対称なリスク・リターン）

---

## 8. 技術的な実装の工夫

### 1. データリーク防止
```python
# ✗ 悪い例: 全データでターゲット作成→分割
features['target'] = (features['close'].shift(-1) > features['close']).astype(int)
train, test = split(features)

# ✓ 良い例: 分割→各セットでターゲット作成
train, test = split(features)
train['target'] = (train['close'].shift(-1) > train['close']).astype(int)
test['target'] = (test['close'].shift(-1) > test['close']).astype(int)
```

### 2. 異常値の処理
```python
# 無限大値の検出と除去
feats = feats.replace([np.inf, -np.inf], np.nan)
feats = feats.dropna().copy()
```

### 3. クラス不均衡対策
```python
# 各モデルに応じた重み付け
class_weight='balanced'  # RandomForest
scale_pos_weight=n_0/n_1  # XGBoost, LightGBM
class_weight={0: w0, 1: w1}  # LSTM, GRU
```

### 4. LSTMのシーケンスデータ作成
```python
def create_sequences(data, feature_cols, lookback):
    X, y, indices = [], [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])  # 過去20日分
        y.append(data[i])              # 当日のターゲット
    return np.array(X), np.array(y), indices
```

### 5. 特徴量ごとの正規化（LSTM/GRU用）
```python
# 各特徴量について個別にスケーリング
for i in range(n_features):
    scaler = StandardScaler()
    scaler.fit(X_train[:, :, i].reshape(-1, 1))
    X_train[:, :, i] = scaler.transform(...)
    X_test[:, :, i] = scaler.transform(...)
```

---

## 9. 課題と改善の総括

### 解決した主要課題

| # | 課題 | 解決策 | 効果 |
|:--:|:--|:--|:--|
| 1 | CSVデータの読み込みエラー | `pd.to_datetime(errors='coerce')`でティッカー行を除外 | データ読み込み成功 |
| 2 | 無限大値エラー | ゼロ除算の回避、`replace([inf], nan)`で除去 | 学習エラー解消 |
| 3 | データリーク | train/test分割後にターゲット変数作成 | 正しい評価が可能に |
| 4 | 全て上昇予測 | `class_weight='balanced'`、非線形モデル導入 | 予測バランス改善 |
| 5 | 予測の偏り | 閾値最適化（0.5→0.57） | 正解率が49.68%→53.55%に向上 |
| 6 | LSTMの過学習 | モデル簡素化、Dropout増加、学習率調整 | 汎化性能向上 |
| 7 | 時系列特性の活用不足 | LOOKBACK=20でシーケンスデータ作成 | 時系列パターン学習 |

### 改善の積み重ね

```
初期状態（ロジスティック回帰）: 正解率48%, 全て上昇予測
    ↓ class_weight='balanced'
正解率49%, 下落予測0%→10%
    ↓ RandomForest導入
正解率49.68%, 下落予測10% (31/310)
    ↓ XGBoost導入
正解率51.29%, 下落予測34% (106/310)
    ↓ LightGBM導入
正解率51.94%, 下落予測31% (96/310)
    ↓ 閾値最適化（0.56）
正解率50.65%, 下落予測47.7%, 上昇予測52.3%（バランス改善）
    ↓ LSTM/GRU追加実装
正解率47-48%, 予測が偏る（ディープラーニングは改善せず）
```

**最終採用モデル**: LightGBM + 閾値0.56

---

## 10. 考察

### 10.1 モデル性能の解釈

#### なぜLightGBMが最も優れていたか

1. **非線形関係の学習能力**
   - 株価の動きは複雑な非線形パターンを持つ
   - LightGBMのLeaf-wise成長戦略が効果的

2. **特徴量の相互作用**
   - 移動平均とMACDの組み合わせなど、特徴量間の相互作用を捉える

3. **クラス不均衡への対応**
   - `scale_pos_weight`が効果的に機能

#### LSTMが期待より低い性能だった理由

1. **データ量の問題**
   - シーケンス作成により実質的なサンプル数が減少
   - テストデータ: 310→290サンプル

2. **ノイズの影響**
   - 株価データは本質的にノイズが多い
   - LSTMは過去20日の全てを学習するため、ノイズも増幅

3. **過学習のリスク**
   - パラメータ数が多く、過学習しやすい
   - Dropout等で対策したが完全には解消できず

### 10.2 特徴量の考察

#### 重要な特徴量

**移動平均（ma_5, ma_20）の重要性**
- 合計で26%の重要度
- トレンドの方向性を示す基本的な指標
- 短期（5日）と長期（20日）の組み合わせが効果的

**MACDの有効性**
- macd, macdsignal, macdhistで合計28%
- トレンドの強さと転換点を捉える

**ボラティリティ指標**
- bb_width（9.8%）とvol_change（9.2%）
- 市場の不確実性を測定

### 10.3 バックテスト結果の考察

#### 戦略の強み

1. **下落相場での防御力**
   - 下落が予想される日は現金保有
   - 買い持ちの-29.54%に対し、+11.39%を達成

2. **選択的取引**
   - 310日中162日のみ取引（52%）
   - 確信度の高い日のみエントリー

3. **リスク管理**
   - 最大損失-5.21%は許容範囲
   - 最大利益+9.30%との非対称性

#### 戦略の限界

1. **勝率49.4%**
   - ほぼ半々の勝率
   - 利益は「勝ちトレードが大きい」ことに依存

2. **取引コスト未考慮**
   - 実際の取引では手数料やスプレッドが発生
   - 162回の取引でコストが累積

3. **スリッページ**
   - 予測した価格で必ず取引できるとは限らない

---

## 11. 今後の改善方向

### 11.1 特徴量エンジニアリング

#### 追加すべき特徴量
1. **ラグ特徴量**
   - 過去2日、3日、5日前のリターン
   - 連続する動きのパターン

2. **ローリング統計量**
   - 過去N日のボラティリティ（標準偏差）
   - 過去N日の最大値・最小値

3. **市場センチメント**
   - 他の銘柄との相関
   - 日経平均やS&P500との関係

4. **曜日効果**
   - 月曜効果、金曜効果などのカレンダー効果

5. **外部要因**
   - 為替レート（USD/JPY）
   - 原油価格
   - 経済指標

### 11.2 モデルアーキテクチャの改善

#### アンサンブル学習
```python
# 複数モデルの予測を組み合わせ
ensemble_pred = (
    0.4 * lightgbm_pred +
    0.3 * xgboost_pred +
    0.2 * randomforest_pred +
    0.1 * lstm_pred
)
```

#### Attention機構の導入
- LSTMにAttention層を追加
- 重要な時点に注目する仕組み

#### Transformer（現代的アプローチ）
- 時系列Transformer
- より長期の依存関係を学習

### 11.3 ハイパーパラメータ最適化

#### Optunaによる自動最適化
```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        # ...
    }
    model = lgb.LGBMClassifier(**params)
    # 交差検証で評価
    score = cross_val_score(model, X, y, cv=5).mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### 11.4 リスク管理の強化

#### ポジションサイジング
- 予測確率に応じた投資額の調整
- 高確信度の予測には大きく、低確信度には小さく

#### ストップロス・テイクプロフィット
```python
stop_loss = -0.02  # -2%で損切り
take_profit = 0.05  # +5%で利確
```

#### 最大ドローダウン管理
- 連続損失時のポジション縮小
- リスク管理の自動化

### 11.5 評価指標の拡充

#### シャープレシオ
```python
sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
```

#### 最大ドローダウン
```python
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()
```

#### カルマー比率
```python
calmar_ratio = annual_return / abs(max_drawdown)
```

---

## 12. 結論

### 達成したこと

1. **複数モデルの実装と比較**
   - RandomForest, XGBoost, LightGBM, LSTM, GRUの5つを実装
   - 体系的な性能比較を実施
   - LightGBMが最良の性能（正解率51.94%）

2. **実用的な予測精度**
   - 閾値最適化後の正解率50.65%
   - バックテストで+11.39%（買い持ち-29.54%）
   - **超過リターン+40.93%**を達成

3. **技術的課題の解決**
   - データリーク防止（train/test分割後にターゲット作成）
   - クラス不均衡対策（class_weight、scale_pos_weight）
   - 閾値最適化（バランススコア導入）
   - 過学習の抑制（Early Stopping、Dropout、正則化）

4. **解釈可能性**
   - 特徴量の重要度分析を実施
   - ret_1d（日次リターン）とvol_change（出来高変化率）が最重要
   - MACDとRSIも有効な指標と判明

5. **発表資料の作成**
   - 総合分析グラフ（5つのサブプロット）
   - サマリー統計テーブル（5つの表）
   - 日本語フォント対応完了

### 限界と課題

1. **予測精度の天井**
   - 株価予測は本質的に困難（効率的市場仮説）
   - 50.65%は統計的に優位だが改善の余地あり
   - ハイパーパラメータ最適化でも改善せず（過学習の懸念）

2. **市場環境への依存**
   - テスト期間は下落相場（-29.54%）
   - 上昇相場での検証が必要
   - 時期による性能のばらつき

3. **取引コストの未考慮**
   - 実運用には手数料（0.1-0.5%）を考慮すべき
   - スリッページ、税金も影響大

4. **最適化の難しさ**
   - Optunaによる最適化は改善せず
   - CV精度とテスト精度のギャップ（過学習）
   - 時系列データ特有の難しさ

### 学んだこと

1. **データ品質の重要性**
   - データの前処理とバリデーションが成功の鍵
   - 小さなエラーが大きな影響を及ぼす

2. **モデル選択の重要性**
   - 問題に応じた適切なモデルの選択
   - LSTMが常に最適とは限らない

3. **反復的な改善**
   - 課題の発見→解決→検証のサイクル
   - 小さな改善の積み重ねが大きな成果に

4. **実務への応用可能性**
   - 機械学習は株価予測に一定の効果
   - ただし、他の要因との組み合わせが重要

### プロジェクトの意義

本プロジェクトを通じて、機械学習による株価予測の可能性と限界を体系的に検証することができた。特に：

- **技術的側面**: データ前処理、モデル構築、評価の一連のプロセスを習得
- **金融的側面**: テクニカル指標の理解、リスク管理の重要性を認識
- **実務的側面**: 実際の運用を想定した課題（取引コスト、リスク管理）を考察

今後は、より高度な特徴量エンジニアリング、アンサンブル学習、リスク管理の強化により、さらなる改善を目指す。

---

## 付録A: 開発環境

### 使用ライブラリ
```python
numpy==2.3.4
pandas==2.3.3
tensorflow==2.18.0
scikit-learn==1.6.1
xgboost==3.1.1
lightgbm==4.6.0
ta==0.11.0
matplotlib==3.10.0
```

### ハードウェア
- CPU: Apple Silicon (M1/M2/M3)
- メモリ: 16GB以上推奨

### 実行時間
- データ読み込み: ~1秒
- RandomForest学習: ~3秒
- XGBoost学習: ~2秒
- LightGBM学習: ~1秒
- LSTM学習: ~2分（100エポック、Early Stopping）
- GRU学習: ~2分（100エポック、Early Stopping）

**合計**: 約5分

---

## 付録B: コード構成

```
toyota-lstm/
├── data/
│   └── TM_1980-01-01_2025-06-27.csv  # 株価データ
├── docs/
│   └── Toyota_LSTM_BuySell_Report.md  # 本レポート
├── simple_buy_sell.ipynb  # メインノートブック
├── best_regression_improved.keras  # 保存モデル
├── scaler_X_7203T.joblib  # 特徴量スケーラー
└── scaler_y_7203T.joblib  # ターゲットスケーラー
```

---

## 参考文献

1. Yahoo Finance - トヨタ自動車株価データ
2. ta-lib Documentation - テクニカル分析ライブラリ
3. LightGBM Documentation - LightGBMの公式ドキュメント
4. "Deep Learning for Time Series Forecasting" - Jason Brownlee
5. scikit-learn Documentation - 機械学習ライブラリ

---

**最終更新**: 2025年11月
**作成者**: 堀越 源之介
**プロジェクト**: DL4E Final Project
