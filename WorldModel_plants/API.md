# API Schema — 2D ゴミ焼却炉シミュレータ

**担当**: Mitsuki Gotou / **対象**: NEDO RSI デモ
**ステータス**: `v0.2` — 実装と整合済み・レビュー依頼中（@Rafael_Engineer @Dai_Engineer）
**最終更新**: 2026-09-14

---

## 0. 3行まとめ

1. シミュレータは `autoresearch` の **`prepare.py` の位置**に収まる。**AI Agent はシミュレータに触らない**（`train.py` だけを編集する）
2. 我々が提供するのは「**データを吐く装置**」と「**単一スカラー評価指標 `val_nrmse`**」の2つだけ
3. **インターフェースはこのドキュメントで凍結する。** 物理の中身は後から差し替わるが、**入出力形式は変えない**

---

## 1. Rafael の論点への回答

### Q1. Action が何になるのか

> **A. 第1段階の action は policy の出力ではなく、データ生成時に外部から与える制御入力シーケンスです。**

`autoresearch` は予測タスク（`val_bpb` 最小化）であって制御タスクではありません。同様に、我々の World Model も学ぶのは

```
(q_t, a_t) -> q_{t+1}
```

だけです。`a_t` はランダム or スクリプト化された入力系列としてシミュレータに与え、データに記録します。**policy 学習 / RL は Future Work** で、第1段階には不要です。

action の中身（人間可読形式）:

| key | 型 | 範囲 | 単位 | 意味 |
|---|---|---|---|---|
| `stoker_speed` | float | 0.01 – 0.30 | m/s | ストーカ（火格子）搬送速度 |
| `waste_feed` | float | 0.5 – 3.0 | kg/s | ごみ投入量 |
| `primary_air` | float[n_zones] | 0.0 – 2.0 | Nm3/s | 一次空気（ゾーン別、ストーカ下部） |
| `secondary_air` | float[n_nozzles] | 0.0 – 1.5 | Nm3/s | 二次空気（ノズル別、炉上部側壁） |

### 可変長問題の解決（重要）

プラントごとに `n_zones` / `n_nozzles` が異なります。`[バルブ1, バルブ2, バルブ3]` のような**設備番号ベースのベクトルにすると、ゾーン数が違うプラントで即座に壊れます**。

そこでシミュレータ内部で**空間場にエンコード**します:

```
a(x, y, t) = sum_j  u_j(t) * b_j(x, y)      # b_j = アクチュエータ j の位置マスク
```

**NN が受け取るのは常に固定形状 `[H, W, 3]`** です:

| ch | 内容 |
|---|---|
| 0 | `primary_air` 場 |
| 1 | `secondary_air` 場 |
| 2 | `fuel_feed` 場（ごみ投入） |
| 3 | `stoker_speed` 場 |

> **v0.1 からの変更（3ch → 4ch）**: 当初 3ch としていましたが、`stoker_speed` が
> どのチャンネルにも乗らず、`a` だけを見る NN から最重要の制御量が見えなくなるため
> 独立チャンネルにしました。**ch 0-2 の意味と index は変えていません。**

> 「第3空気口」ではなく「**この座標から、この向きに、これだけ空気が入る**」として表現する。
> これがプラント非依存化の鍵で、後から Plant B / C を足しても NN 側の変更が不要になります。

---

### Q2. 報酬は？

> **A. 第1段階では不要です（予測タスクなので）。ただし計算して出力に含めておきます。**

将来 policy / MPC を足すときにデータを取り直さずに済むよう、`reward` を毎ステップ計算して `.npz` に入れておきます。**第1段階では誰も使いません。**

```
r_t = - w_T   * (T_exit - T_target)^2
      - w_O   * (Y_O2_exit - Y_O2_target)^2
      - w_CO  * CO_proxy
      - w_unb * Unburned
      - w_E   * E_input
```

デフォルト: `T_target = 1123 K (850 degC, 法規準拠)`, `Y_O2_target = 0.06`

**proxy であることの明示（重要 / 資料にもそう書いてください）**

一段反応モデルには **CO も NOx も存在しません**。以下は代理指標です:

| 指標 | 定義 |
|---|---|
| 未燃分 `Unburned` | 出口側の `integral Y_F dV` |
| CO 代理 `CO_proxy` | 局所当量比 `phi > 1` の領域体積 |
| NOx 代理 `NOx_proxy` | `T > 1500 K` の滞留時間積分 |

専門家レビューで必ず突かれる箇所なので、最初から proxy と名乗っておくのが安全です。

---

### Q3. ゴミ焼却の toy model っぽくできるか

> **A. できます。2D 気相燃焼 + 1D ストーカ bed モデルの結合で作ります。**

ストーカ炉の本質は、ごみ層（bed）の **乾燥 → 熱分解（揮発分放出）→ チャー燃焼 → 灰** という進行と、その上のガス相燃焼の**二層構造**です。気相だけだと**一次空気ゾーン制御に意味が出ず**、「ストーカ炉」を名乗れません。

bed は 1D（ストーカ方向のみ）なので**計算コストはほぼゼロ**ですが、これで
着火 / 火炎の移動 / 酸素不足 / ベルト速度による燃え残り / 温度暴走
が全部表現できます。

---

### Q4. 評価指標

> **A. `autoresearch` の `val_bpb` に相当する単一スカラー `val_nrmse` を提供します。低いほど良い。**

```
val_nrmse = mean over (hold-out episodes, channels) of
              RMSE( H-step open-loop rollout ) / std(channel)
```

- `H = 16` ステップの開ループロールアウト（1ステップ予測だけだと長期安定性が測れないため）
- チャンネルごとの標準偏差で正規化（温度 1000 K と質量分率 0.1 のスケール差を消す）
- **これ1本**なので、`autoresearch` の「反復するたびに下がっていく図」がそのまま出ます

`integration/evaluate.py` が stdout に1行で出します:
```
val_nrmse: 0.0834
```

---

## 2. API Schema

3レベルで提供します。**`autoresearch` は Level 1 だけで動きます。** Level 2/3 は将来のオンライン学習用。

### Level 1: CLI（オフラインデータセット生成）

```bash
python -m wmf.sim.run \
    --plant   configs/plants/plant_a.yaml \
    --actions random \
    --steps   400 \
    --save-every 4 \
    --seed    0 \
    --out     data/train/ep_0001.npz
```

`--actions` は `random` / `scripted` / JSON ファイルパス のいずれか。

### Level 2: Python API（gym-like）

```python
from wmf.sim import IncineratorEnv

env = IncineratorEnv(config="configs/plants/plant_a.yaml", seed=0)
obs = env.reset()                      # dict: q, a, g, bed

action = {
    "stoker_speed":  0.18,
    "waste_feed":    1.2,
    "primary_air":   [0.8, 1.1, 0.9, 0.5, 0.3],
    "secondary_air": [0.6, 0.6, 0.4, 0.4, 0.3, 0.3],
}
obs, reward, info = env.step(action)
```

### Level 3: batched JAX API（大量データ生成）

```python
from wmf.sim import rollout_batch

# プラントを vmap でバッチ並列実行（格子形状が全プラント共通なので可能）
states = rollout_batch(configs, action_seqs, n_steps)   # [B, T, H, W, 7]
```

---

## 3. データ形式（`.npz`、エピソード単位）

**格子は全プラント固定 `H = W = 64`。** 幾何の違いは `mask` と `sdf` で表現します。
配列形状が揃うので (a) `vmap` でプラント並列にでき、(b) NN の入力形状も固定されます。

| key | shape | dtype | 内容 |
|---|---|---|---|
| `q` | `[T, 64, 64, 7]` | float16 | 状態場 |
| `a` | `[T, 64, 64, 4]` | float16 | アクチュエータ場（空間エンコード済み） |
| `g` | `[64, 64, 4]` | float32 | 幾何（時間不変） |
| `bed` | `[T, 64, 3]` | float32 | 1D bed 状態 |
| `reward` | `[T]` | float32 | 上記 `r_t`（第1段階では未使用） |
| `proxies` | `[T, 6]` | float32 | `t_exit, o2_exit, yf_exit, co_proxy, nox_proxy, unburnt_bed` |
| `a_scalar` | `[T, D]` | float32 | 生の action 値（人間可読、D はプラント依存） |
| `meta` | JSON string | — | 下記 |

### `q` のチャンネル定義

| ch | 記号 | 単位 | 実測レンジ | 実測 p1–p99 |
|---|---|---|---|---|
| 0 | `T` 温度 | K | 380 – 1700 | 450 – 1520 |
| 1 | `u` x方向速度 | m/s | -17 – 18 | -4.8 – 3.4 |
| 2 | `v` y方向速度 | m/s | -13 – 13 | -2.8 – 2.2 |
| 3 | `p` 圧力（ゲージ） | Pa | -27 – 195 | 0 – 26 |
| 4 | `Y_F` 燃料（揮発分）質量分率 | – | 0 – 0.77 | 0 – 0.65 |
| 5 | `Y_O2` 酸素質量分率 | – | 0 – 0.232 | 0 – 0.232 |
| 6 | `Y_P` 生成物質量分率 | – | 0 – 0.30 | 0 – 0.30 |

> **v0.1 の値域は実機の数値を書いたもので、実測と食い違っていました。** 上表は
> Plant A/B/C x ランダム action の**実測値**です。正規化にはこちらを使ってください。

### `g` のチャンネル定義

| ch | 内容 |
|---|---|
| 0 | `mask`: 1 = 流体領域, 0 = 壁・固体 |
| 1 | `sdf`: 壁までの符号付き距離（正規化済み） |
| 2 | `actuator_layout`: アクチュエータ位置の静的マップ |
| 3 | `sensor_mask`: 温度計・酸素計の設置位置 |

### `bed` のチャンネル定義

| ch | 内容 |
|---|---|
| 0 | `m_w` 水分 |
| 1 | `m_v` 揮発分 |
| 2 | `m_c` チャー |

### `meta`（JSON）

```json
{
  "plant_id": "plant_a",
  "plant_hash": "a3f9c1...",
  "dx": 0.1875,
  "dy": 0.1094,
  "dt": 0.002,
  "save_every": 4,
  "n_zones": 5,
  "n_nozzles": 6,
  "dimensionless": {"Re": 12400.0, "Da": 3.2, "Pe": 8900.0, "Ri": 0.45, "phi": 0.85},
  "norm": {"q_mean": [...], "q_std": [...]},
  "a_scalar_keys": ["stoker_speed", "waste_feed",
                    "primary_air_0", ..., "secondary_air_5"]
}
```

### `dx` を必ず使ってください（重要）

炉幅 8 m と 15 m を**同じ 64x64 格子に載せる**ため、**物理的な格子幅 `dx` がプラントごとに違います**。
NN はこれを知らないと「この炉は大きい」と「この炉は細かく刻まれている」を区別できず、**プラント横断で汎化しません**。

対策として、`meta.dx` を渡すのに加えて、**`train.py` 側で `dx` を定数チャンネルとして入力に足すこと**を推奨します（1行で足せます）。
将来 Plant B / C を混ぜた学習をする段階で効いてきます。

---

## 4. プラント設定（YAML）

```yaml
plant_id: plant_a
furnace:
  width:  12.0      # m
  height:  7.0      # m
stoker:
  length:      12.0  # m
  inclination:  8.0  # deg
  zones:        5
primary_air:
  zones: 5           # ストーカ下部に等分割配置
secondary_air:
  nozzles: 6         # 炉上部側壁
flue:
  diameter_ratio: 0.25    # 炉幅比
  position: top_right
waste:
  moisture:      0.35     # 含水率
  lhv:           9.0      # MJ/kg 低位発熱量
  volatile_frac: 0.60
  ash_frac:      0.10
wall:
  heat_transfer: 25.0     # W/m2K
```

Plant A / B / C を用意し、形状・ゾーン数・煙道位置を変えます。

| | 位置づけ | 設計 |
|---|---|---|
| Plant A | メインの学習対象 | 標準的な炉、一次空気5ゾーン |
| Plant B | hold-out 候補 | 長い炉、6ゾーン、煙道位置が異なる |
| Plant C | hold-out 候補 | 傾斜ストーカ、二次空気配置が異なる |

> **幾何情報 (`g`, `meta.dimensionless`) は常にデータに含めます。**
> World Model 側でそれを使うかどうかは **`train.py` 側（= AI Agent）の裁量**です。
> 第1段階で Plant A のみ使う場合でも、この形式なら後から B/C を足すのに**シミュレータ側の変更は不要**です。

---

## 5. autoresearch への組み込み

`integration/` 以下に、そのまま使える形で置きます。

| ファイル | 役割 |
|---|---|
| `integration/prepare.py` | `.npz` 群を読み、正規化して学習可能な配列にする（**autoresearch の `prepare.py` に相当・不変**） |
| `integration/train_baseline.py` | **AI Agent が編集する出発点**。素朴な CNN。意図的に弱くしてあります |
| `integration/evaluate.py` | `val_nrmse` を単一スカラーで出力 |
| `integration/program_draft.md` | `program.md` のシミュレータ側ドラフト |

### ベースラインは意図的に弱くしてあります

AI Agent が改善できる余白がないと `autoresearch` のデモが成立しないため、`train_baseline.py` は
小容量・短学習・素朴な L2 損失・1ステップ予測のみ、にしてあります。

**Agent が改善できる余地（想定）**: 残差予測への変更 / unrolled training / チャンネル正規化 /
U-Net 化 / `dx` や幾何チャンネルの活用 / 発散ペナルティの追加 / データ拡張 / 学習率スケジュール。

---

## 6. 現在のステータスと予定

| Day | 内容 | 状態 |
|---|---|---|
| Day 1 | **この API.md**（schema 確定） | 作業中 |
| Day 2 | **ダミー物理のスタブ + サンプルデータセット**（train 200 / val 50） | 予定 |
| Day 3–5 | 気相ソルバ本実装（人工圧縮性 + 浮力 + 一段反応） | 予定 |
| Day 6–7 | 1D bed 結合 / Plant A・B・C / ゴールデンテスト | 予定 |
| Day 8–9 | 本番データ生成 + `integration/` 一式 | 予定 |
| Day 10 | 可視化・デモ通し・バッファ | 予定 |

> **Day 2 の時点で、上記フォーマット通りの `.npz` が repo に入ります。**
> 物理はダミー（移流 + 拡散のみ）ですが**形式は本番と完全に同一**なので、
> Dai / Rafael はそこから `autoresearch` 接続作業を並行して開始できます。
> Day 5 以降にファイルを差し替えても、**読み込みコードの変更は不要**です。

---

## 7. 変更管理

- **この schema は Day 1 で凍結します。** 以降の変更は Slack で事前合意してから
- 物理モデルの精緻化は `q` の値を変えますが、**shape / key / 単位は変えません**
- チャンネル追加が必要になった場合は**末尾に追加**し、既存 index は動かしません

## 8. レビューしてほしい点

1. `action` の項目と範囲はこれで妥当か（特に `secondary_air` を入れるか）
2. `val_nrmse` の `H = 16` ステップは妥当か。`autoresearch` 側で扱いやすい長さか
3. `.npz` エピソード単位で良いか、それとも1つの大きな配列に結合した方が `prepare.py` を書きやすいか
4. Day 2 のスタブに必要なエピソード数（暫定 train 200 / val 50）
5. 実データ CSV（`200801-0831.csv`）のカラム構成 — **出力変数名を寄せておくと将来の接続が楽になる**ので、カラム一覧だけ共有いただけると助かります
