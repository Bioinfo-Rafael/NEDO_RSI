# インターフェース仕様

3つの層が何を受け取り、何を返すか。数値はすべて実装から取っています
（`src/wmf/`, `control_research/`）。物理側・制御側を別々に実装できるよう、
層の境界をテンソルの形と単位で定義しています。

```
        目標 g                      操作 u                      状態
          │                           │                          │
          ▼                           ▼                          │
   ┌─────────────┐   予測    ┌──────────────┐            ┌───────────────┐
   │ Controller  │◄─────────│ World Model  │            │  Simulator    │
   │  C (MPC)    │          │      M       │            │  = プラント    │
   └──────┬──────┘          └──────────────┘            └───────┬───────┘
          │                         ▲                           │
          │ u（最初の1手のみ）        │  学習データ                 │ 計測 y
          └─────────────────────────┼──────────────────────────►│
                                    └───────────────────────────┘
```

**役割分担**: `M` は「何が起こるか」、`C` は「何をすべきか」、Simulator は
「真値」。`C` は Simulator を一切見ず、`M` の中だけで未来を試して操作を決めます。

---

## 0. 共通の時間刻み

| 量 | 値 | 意味 |
|---|---|---|
| `dt` | **1.0 ms** | 気相ソルバの積分刻み |
| `save_every` | **100** | 1制御ステップ = 100積分ステップ |
| **制御周期** | **0.1 s** | `C` が操作を更新する間隔 |
| エピソード長 | 250〜300 ステップ = **25〜30 s** | 火格子滞留 16.7 s より長く取る必要がある |

> **なぜ 25 s 以上か**: ごみが火格子を渡り切るのに 16.7 s かかります。これより
> 短いエピソードでは `waste_feed` の効果が出る前に終わり、実測で未燃分が小数3桁
> まで動きませんでした。**むだ時間より短い実験は、効かない操作端をデータに混ぜます。**

---

## 1. Simulator（プラント／真値）

### 1.1 入力

**(a) 設備定義** — `configs/plants/plant_*.yaml` → `PlantConfig`

```yaml
plant_id: plant_a
furnace:   {width: 12.0, height: 7.0}          # m
stoker:    {length: 12.0, inclination: 8.0, zones: 5}   # m, deg, 段数
primary_air:   {zones: 5}
secondary_air: {nozzles: 6}
flue:      {diameter_ratio: 0.25, position: top_right}
waste:     {moisture: 0.35, lhv: 9.0, volatile_frac: 0.60, ash_frac: 0.10}
           # volatile_frac / ash_frac は乾燥基準。char = 1 - volatile - ash
wall:      {heat_transfer: 25.0, temperature: 600.0}    # W/m2K, K
```

これが `SimParams.from_plant()` で以下に変換されます（**設備ごとに異なる**）:

| 導出量 | 式 | plant_a | plant_b | plant_c |
|---|---|---|---|---|
| `q_cp` [K] | `lhv × 1e6 / c_p` | 7826 | 6522 | 9565 |
| `lam_loss` [1/s] | `h_wall / (ρ₀ c_p H)` | 0.00887 | 0.01242 | 0.00552 |
| `t_wall` [K] | そのまま | 600 | 580 | 620 |

**(b) 操作 `action`** — 制御周期ごとに1つ

| キー | 型 | 範囲 | 単位 | レート制約 |
|---|---|---|---|---|
| `stoker_speed` | float | 0.010 – 0.030 | m/s | ±0.0010 /step |
| `waste_feed` | float | 1.0 – 1.6 | kg/s | ±0.06 /step |
| `primary_air` | float[`n_zones`] | 1.0 – 1.5 | Nm³/s | ±0.015 /step |
| `secondary_air` | float[`n_nozzles`] | 0.2 – 0.6 | Nm³/s | ±0.018 /step |

> **範囲は基準プラント（plant_a: 火格子 12 m、LHV 9 MJ/kg）の値で、プラントごとに火格子長と
> 発熱量でスケールします**（`wmf.actions.envelope(cfg)`）。実機の投入機とファンは火格子に合わせて
> 設計されるので、火格子面積あたりの熱負荷と空気量を一定に保ちます（投入 ∝ L/LHV、空気 ∝ L、
> よって λ は保存）。plant_b は投入 1.5–2.4 kg/s・一次 1.25–1.875、plant_c は 0.61–0.98・0.75–1.125。
>
> **基準値は実測で決めました。** 火格子速度は 12 m のストーカ上で滞留 400–1200 s
> となる値（実機ログの 660–720 s を挟む）。空気は 15 点の掃引で炉が着火を保つ領域で、
> その中で炉温 714–1295 K、O₂ 6.5–15.9%、灰未燃 3.5–7.9% と制御に十分な幅があります。
> 一次空気を 1.0–1.5 まで絞ると層のチャーが空気不足で燃えず、層温度が 330 K に落ちて
> 投入の 58% が未燃のまま火格子から落ちる — 実機と同じ失敗モードで、包絡線の外側です。
> レート制約は各軸とも自分のスパンの約 3%/s、すなわちフルストロークに 30–60 s。
> 実機の空気ダンパの動作時間に合わせています。

> **可変長の扱い**: ゾーン数・ノズル数はプラントごとに違います（4/6/8）。
> 設備番号のベクトルにすると別プラントで壊れるため、内部では**空間場**に変換します:
>
> $$a(x,y,t)=\sum_j u_j(t)\,\frac{b_j(x,y)}{\int b_j\,dA}$$
>
> 面積で積分すると元のスカラーが**厳密に**戻ります（テストで担保）。NN が受け取るのは
> 常に `[64,64,4]` で、`primary_air / secondary_air / fuel_feed / stoker_speed`。

### 1.2 計算するもの

64×64 固定格子上で、以下を 100 積分ステップぶん時間発展させます。

**気相**（人工圧縮性 + Boussinesq 浮力 + 一段 Arrhenius 反応）

$$
\frac{\partial p}{\partial t}=-c^2\rho_0(\nabla\cdot\mathbf u-f_{\rm inj}),\qquad
\frac{\partial \mathbf u}{\partial t}=-(\mathbf u\cdot\nabla)\mathbf u-\frac{\nabla p}{\rho_0}+\nu_t\nabla^2\mathbf u+g\frac{T-T_{\rm ref}}{T_{\rm ref}}\hat y-C_D\mathbf u
$$

$$
\frac{\partial T}{\partial t}+\mathbf u\cdot\nabla T=\alpha_t\nabla^2T-\lambda_{\rm loss}(T-T_{\rm wall})+S_{\rm char},\qquad
\dot\omega=A\,Y_F Y_{O_2}e^{-E_a/RT}
$$

**1D ストーカ bed**（火格子方向64点、乾燥→熱分解→チャー燃焼 + **層温度** $T_b$）

$$
\frac{\partial}{\partial t}\begin{bmatrix}m_w\\ m_v\\ m_c\end{bmatrix}
+v_{\rm eff}\frac{\partial}{\partial x}\begin{bmatrix}m_w\\ m_v\\ m_c\end{bmatrix}
=\begin{bmatrix}-R_{\rm dry}\\ -R_{\rm pyro}\\ -R_{\rm char}\end{bmatrix}+\dot m_{\rm feed}\boldsymbol\chi
$$

$$
(m_w{+}m_v{+}m_c)c_s\frac{\partial T_b}{\partial t}
= \underbrace{\varepsilon\sigma\big(\overline{T^4}-T_b^4\big)}_{\text{炉からの放射}}
+ \underbrace{R_{\rm char}Q_{\rm char}}_{\text{チャー燃焼}}
- \underbrace{\dot m_{\rm air}c_p (T_b-T_{\rm ph})}_{\text{一次空気の予熱}}
- \underbrace{R_{\rm dry}L_v + R_{\rm pyro}H_{\rm pyro}}_{\text{潜熱}}
+ \; (\text{火格子方向の移流と火炎伝播})
$$

乾燥・熱分解・チャー燃焼のゲートはすべて $T_b$ で駆動します。ガス温度で駆動すると、
一次空気が入るのがまさにその位置なので、実時定数にした途端に炉が消えます。

**実機に合わせるために必要だった要素**（いずれも欠けていると炉が消えます）:

| 要素 | 値 | なぜ必要か |
|---|---|---|
| 一次空気予熱器 | $T_{\rm ph}=450$ K (177 °C) | これが無いと層収支が $-210$ kW/m² |
| 放射は $\overline{T^4}$ で平均 | — | 薄い火炎(1900 K)を持つ場で $T$ を先に平均すると放射が 7 倍過小 |
| 放射キャビティ（着火アーチ） | 炉全体の平均 | 柱ごとだと供給端が暗く、新しいごみが着火しない |
| 層内の火炎伝播 | $D_{\rm bed}=1.5\times10^{-3}$ m²/s | 着火前線が火格子と逆向きに 1–10 mm/s で遡る |
| 二次空気ジェットの貫通 | 炉幅の 35% | 1セルに集中すると壁沿いの冷たいカーテンで火炎が分断される |
| 下部炉の耐火ライニング | $T_{\rm wall}=1250$ K, 熱損失 0.1倍 | 炉の連続着火源。上部は水管壁 600 K |

bed から出る揮発分が気相の燃料源になります（火格子直上4セルに重み
`(0.4,0.3,0.2,0.1)` で分散）。チャー燃焼熱は層内に残り、気相へは**層温度まで予熱された
一次空気**として伝わります。

### 1.3 出力

| 名前 | 形 | dtype | 内容 |
|---|---|---|---|
| `q` | `[T, 64, 64, 7]` | float16 | `T[K], u[m/s], v[m/s], p[Pa], Y_F, Y_O2, Y_P` |
| `a` | `[T, 64, 64, 4]` | float16 | 操作場（上記） |
| `g` | `[64, 64, 4]` | float32 | `mask, sdf, actuator_layout, sensor_mask`（時間不変） |
| `bed` | `[T, 64, 3]` | float32 | `m_w, m_v, m_c` [kg/m²] |
| `proxies` | `[T, 6]` | float32 | `t_exit, o2_exit, yf_exit, co_proxy, nox_proxy, unburnt_bed` |
| `reward` | `[T]` | float32 | 下記 |
| `meta` | JSON | — | `dx, dy, dt, 無次元数(Re,Pe,Ri,Da,φ), 設備ハッシュ` |

**`proxies` の定義**（制御が実際に見る量）

| | 単位 | 意味 |
|---|---|---|
| `t_exit` | K | 煙道出口の平均温度 |
| `o2_exit` | – | 出口 O₂ 質量分率 |
| `yf_exit` | – | 出口の未燃ガス |
| `co_proxy` | – | 局所当量比 φ>1 の領域体積比 |
| `nox_proxy` | – | T>1500 K の領域体積比 |
| `unburnt_bed` | kg/m²/s | **火格子端から落ちる未燃固形分**（＝熱しゃく減量に相当） |

> **`co_proxy` / `nox_proxy` は proxy です。** 一段反応モデルには CO も NOx も
> 存在しません。資料でも必ず proxy と明記してください。

### 1.4 速度

| | |
|---|---|
| 1制御フレーム（100積分ステップ） | **21 ms** |
| 1エピソード（250フレーム = 25 s） | 約 5 s |
| バッチ実行 | `rollout_batch` が `vmap` で**異なる幾何のプラントを同時実行** |

### 1.5 API

```bash
# CLI
python -m wmf.sim.run --plant configs/plants/plant_a.yaml \
    --actions random --steps 250 --seed 0 --out data/train/ep_0001.npz
```
```python
# gym-like
from wmf.sim import IncineratorEnv
env = IncineratorEnv("configs/plants/plant_a.yaml", seed=0)
obs, reward, info = env.step({"stoker_speed":0.18, "waste_feed":2.5,
                              "primary_air":[0.8]*5, "secondary_air":[0.5]*6})
# batched
from wmf.sim.rollout import rollout_batch
q, bed, diag = rollout_batch(q0, bed0, a_seq, statics, prm, save_every, n_frames)
```

---

## 2. World Model `M`（制御用）

**重要**: 制御に使う `M` は**64×64の場を予測しません**。制御に必要なのは数個の
スカラーであり、場のモデルは実測で 1〜2 秒しか信号が持ちません（プラントの
むだ時間 16.7 s に対して不足）。Ha & Schmidhuber 2018 が画素ではなく32次元潜在を
ロールアウトするのと同じ構造です。

### 2.1 入出力

| | 記号 | 形 | 中身 |
|---|---|---|---|
| 観測 | `y` | `[B, K, 4]` | `t_exit, o2_exit, yf_exit, unburnt_bed` |
| 操作 | `u` | `[B, K, 4]` | `stoker_speed, waste_feed, primary_level, secondary_level` |
| 出力 | `ŷ` | `[B, H, 4]` | 将来 H ステップの観測 |

`primary_level` / `secondary_level` はゾーン平均。`C` が出す集約4ノブを、各プラントの
実アクチュエータ数へ展開します。

### 2.2 実装が満たすべき契約

```python
init_params(key)                          -> params        # アンサンブル1体
warm_up(params, y_hist, u_hist)           -> state         # 履歴から内部状態を推定
predict(params, state, y_last, u_future, bias=None) -> [B, H, 4]
train(data, stats, seconds, seed)         -> [params, ...] # アンサンブル
```

モジュール定数 `N_CV=4`, `WARMUP`, `ENSEMBLE` も公開すること。
**この4関数のシグネチャさえ守れば、中身は完全に自由**です（`C` はこれしか呼びません）。

### 2.3 現行ベースライン

| | 値 | 備考 |
|---|---|---|
| 構造 | GRU 1層 | 隠れ状態が**計測できない bed 状態**を履歴から推定する |
| 隠れ層 | 4 | Codex が 48→4 に縮小して性能が上がった |
| アンサンブル | 5体 | `C` が悲観評価と信頼領域に使う |
| 学習地平 | 30ステップ（3.0 s）unrolled | 1ステップ学習では制御に使えない |
| 学習予算 | 150 s 固定 | 実験間で公平にするため |

> **なぜ再帰か**: 次に何が燃えるかを決める「火格子上の燃料分布」は**計測できません**。
> 実炉にも bed を測る計器はありません。`C` が持っているのは出口3計測の履歴だけなので、
> そこから推定するしかない。記憶の無い写像では原理的に不可能です。

### 2.4 未実装（既知のギャップ）

- **設備条件 $\mu_{\rm known}$ を毎ステップ注入していない** — 現在 `predict` は
  設備情報を一切受け取らず、`plant_a` 専用です
- 報酬予測 $R_\theta$・価値予測 $V_\theta$ が無い（損失が観測MSEのみ）
- 部分観測・センサーマスク非対応

---

## 3. Controller `C`（CEM-MPC）

### 3.1 入出力

```python
u = mpc.act(y_hist,   # [K, 4] 直近の計測（物理単位）
            u_hist,   # [K, 4] 直近の操作
            target,   # [4]    目標値 g
            u_prev)   # [4]    現在の操作
# -> u: [4]  次に適用する操作（物理単位、範囲・レート制約を満たす）
```

### 3.2 中で何をしているか

毎制御ステップ（0.1 s）に **256候補 × 30ステップ × 4反復 ≈ 3万本の仮想未来**を
`M` の中でロールアウトし、最良の計画の**最初の1手だけ**を適用します。

$$
u^\*=\arg\min_u\ \underbrace{\max_{k\in\text{ensemble}}\sum\|\hat y^{(k)}-g\|_Q^2}_{\text{悲観評価}}+\underbrace{\lambda\,\mathrm{Var}_k[\hat y^{(k)}]}_{\text{信頼領域}}+\underbrace{\|\Delta u\|_R^2}_{\text{急変抑制}}
$$

| 安全機構 | 実装 |
|---|---|
| 受動的地平 | 最初の1手のみ適用し毎回再最適化 |
| オフセットフリー | $d_t=y^{\rm meas}-y^{\rm pred}$ を EWMA で推定し予測に加算 |
| 悲観評価 | アンサンブル**最悪**メンバーで採点 |
| 信頼領域 | メンバー間の**不一致**にペナルティ（＝学習分布外を避ける） |
| レート制約 | 操作を**増分でサンプリング**するので構成上必ず満たす |

### 3.3 設定

| | 値 |
|---|---|
| 予測地平 | 30ステップ = **3.0 s** |
| 候補数 / エリート / 反復 | 256 / 32 / 4 |
| 計算時間 | **97 ms / 制御ステップ** |

---

## 4. 評価

### 4.1 制御性能（`control_research/evaluate.py`、読み取り専用）

$$
\text{control\_cost}=\frac{1}{|\text{scenarios}|}\sum\Big[\sum_k w_k\frac{\mathrm{IAE}_k}{\sigma_k}+0.5\cdot\text{overshoot}+0.1\cdot\text{move}\Big]
$$

IAE は整定猶予（60ステップ）を除いた期間の絶対誤差積分。各チャンネルは同定データ上の
標準偏差で正規化（K と質量分率を比較可能にするため）。

**測定済みの基準値**

| | control_cost | 何もしない比 |
|---|---|---|
| 何もしない（初期操作を保持） | 1.4359 | 1.00 |
| 未学習 `M`（乱数初期値） | 3.4828 | 2.43 悪い |
| 現行ベースライン | 0.9575 | 0.67 |
| Codex 改良後（4実験） | **0.7916** | **0.55** |

> **重要**: 未学習が壊滅的なので指標は `M` を測っています。しかしそこから先は
> **容量と学習を増やすほど制御が悪化**しました。**予測精度 ≠ 制御性能**です。

### 4.2 目標の到達可能性

目標を決める前に**同時到達できるか**を同定データで確認すること。

| 目標 | 定常データ中の出現率 | 判定 |
|---|---|---|
| T=900 K, O₂=6.0% | 0.54% | ❌ 実質不可能 |
| T=900 K, O₂=4.0% | 7.10% | ✅ |
| T=900 K, O₂=3.0% | 9.21% | ✅ |

---

## 5. 他のエンジニアが差し替えられる単位

| 差し替える対象 | 守る契約 | 影響範囲 |
|---|---|---|
| 物理モデル（燃焼・乱流・輻射） | §1.3 の出力テンソル形と単位 | Simulator 内で閉じる |
| World Model の構造 | §2.2 の4関数 | `C` は無変更 |
| Controller（MPC→RL等） | §3.1 の `act()` | `M` は無変更 |
| 設備の追加 | §1.1 の YAML | 自動で幾何ラスタライズ |

**検証**: `pytest tests/ -q`（183件）。保存則・解析解・action envelope 全16隅 ×
3プラントでの安定性と再現性を検査します。物理を差し替えたら、まずここを通してください。
