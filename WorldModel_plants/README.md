# WorldModel_fluid — ストーカ式焼却炉の物理シミュレータと、その上で動く World Model 制御ループ

このMacでの環境準備・実測した動作確認結果・ローカル修正は[LOCAL_RUN.md](LOCAL_RUN.md)を参照してください。

固定LLMで予測・制御の両方を改善するA/B実験は[research/README.md](research/README.md)から実行できます。Agentの指示は[research/Prompt.md](research/Prompt.md)に集約し、元の実験とは別の作業コピーと固定評価を使います。

NEDO RSI デモ用。**純 Python（JAX）** で書かれた 2D 焼却炉シミュレータと、
それを「実機」に見立てて World Model `M` と Controller `C` を回す一式です。

```
   Simulator (真値の定義)  ──観測──▶  M : World Model (学習・Codex が書き換える)
        ▲                                   │
        └────────── 操作量 ─────────  C : Controller (M の中で計画する MPC)
```

- シミュレータは **Python パッケージ `wmf`**。モデルも Python なので同一プロセスで動き、
  結合は `env.step(action) → obs` の 1 メソッドです（下記「モデルの差し込み方」）。
- 既存の制御実験の入口は **`control_research/run.py`**、課題文は **`control_research/program.md`**。
  今回追加した固定LLMのA/B比較の入口は **`research/run.py`**、課題文は **`research/Prompt.md`**。API サーバ等は不要です。

## 0. 全体像と読む順番

このフォルダには、**共通のSimulator実装、目的の異なる3つの実験入口、データ、実行結果**が同居しています。
フォルダ名は`WorldModel_plants`、Pythonでimportするパッケージ名は`wmf`です。
既存資料の`WorldModel_fluid`という名前は、同じコード系統の以前の表記です。

### まず何を開くか

| やりたいこと | 読む場所 |
|---|---|
| このMacで動かす・準備状況を確認する | [LOCAL_RUN.md](LOCAL_RUN.md) |
| 今回の固定LLM・A/B実験を理解する／再実行する | [research/README.md](research/README.md) |
| 今回の結果・グラフ・動画を見る | [research/RESULTS_20260915.md](research/RESULTS_20260915.md) |
| Codexのプロンプト・作業範囲・ログを調べる | [実験の動作解説](research/EXPERIMENT_WALKTHROUGH_20260915.md) |
| Simulatorを自分のコードから呼ぶ | 本文の「2a」、[src/wmf/furnace.py](src/wmf/furnace.py)、[src/wmf/sim/env.py](src/wmf/sim/env.py) |

### 3つの実験入口の違い

| 場所 | 何を学習・評価するか | 課題文と入口 | 今回との関係 |
|---|---|---|---|
| [integration/](integration/) | 炉内の空間場`q`を予測するWorld Model。予測NRMSEと別プラントへの汎化を評価 | `program.md`、`train.py` | 初期の予測実験。今回のA/B実験では使っていない |
| [control_research/](control_research/) | 計測できる4出力を予測するWorld Modelを学習し、MPCの制御costで評価 | `program.md`、`run.py` | 既存の制御実験。今回の初期モデルや実装の足場 |
| [research/](research/) | 固定LLMでA/B各4回の提案を実行。予測と制御の両方で選択し、最後に未使用条件で評価 | `Prompt.md`、`run.py` | 今回追加・実行した仕組み |

同名の`run.py`・`evaluate.py`・課題文があっても、別の実験です。
今回の再実行は`research/run.py compare`を使います。`make baseline`は既存の`control_research/`を実行します。

```text
configs/plants/ ── 炉の仕様 ──┐
                             ▼
                       src/wmf/                 共通のSimulator・MPC
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  integration/       control_research/        research/
  空間場の予測        4出力のモデル＋制御       今回の固定LLM A/B実験
  data/を使用         data_ctrl_v2/を使用       初期データ・実装をrun内へコピー
         │                   │                   │
         └──── scripts/・figs/など                ▼
                                         research/runs/<run_id>/
                                         コード・追加データ・ログ・評価・動画
```

`data/`は`integration/`が期待する生成先で、このcheckoutには現時点でありません。
`data_multi_v2/`は別プラントの制御用データであり、`integration/`が要求する空間場データの代わりにはなりません。

### リポジトリ直下の各ファイル

| ファイル | 役割・読むときの注意 |
|---|---|
| [README.md](README.md) | リポジトリ全体の入口。この構成案内と既存の実行例 |
| [LOCAL_RUN.md](LOCAL_RUN.md) | このMacでの環境構築、動作確認、実測時間、ローカル修正の記録 |
| [API.md](API.md) | 空間場予測を中心とした初期のAPI設計資料。今回の実験の編集権限・操作範囲を定める文書ではない |
| [INTERFACES.md](INTERFACES.md) | Simulator・World Model・Controller間の形状や単位を説明した既存仕様書。旧時間刻み等の記述が残る |
| [pyproject.toml](pyproject.toml) | Pythonパッケージ`wmf`の依存関係、インストール設定、pytest設定 |
| [Makefile](Makefile) | `make install`・`selfcheck`・`test`・`baseline`等の既存処理の短縮コマンド |
| [.gitignore](.gitignore) | 仮想環境・生成物・ログ等をgit管理から除外する規則 |

資料は開発時期が異なります。例えばルートの`INTERFACES.md`には制御周期0.1秒の記述がありますが、
今回の`research/`は333×0.003=0.999秒です。**今回の仕様は[research/INTERFACES.md](research/INTERFACES.md)、
実際の過去runの仕様はその`reference/`と`manifest.json`を確認してください。**

### リポジトリ直下の各ディレクトリ

| ディレクトリ | 中身・役割 | 今回のA/B実験との関係 |
|---|---|---|
| [src/](src/) | 実装本体。`src/wmf/`以下にSimulator、炉の幾何、操作、MPCを置く | 実行開始時に固定参照へコピー |
| [configs/](configs/) | `plants/plant_a.yaml`・`plant_b.yaml`・`plant_c.yaml`。炉のサイズ、空気口、ごみ性状等 | 今回は`plant_a`を使用 |
| [integration/](integration/) | 空間場の予測モデル開発。`train.py`、固定の`prepare.py`・`evaluate.py`、課題文、既存`results.tsv`・`agent_report.py` | 今回は未使用 |
| [control_research/](control_research/) | 既存のモデル学習・MPC採点、課題文、過去のログ | 初期モデル等の足場。元ファイルは今回のAgentの直接編集先ではない |
| [research/](research/) | 今回追加した提案・生成・学習・採点・ログ解析を管理するコードと資料 | 今回の実験入口。詳細はこの中のREADME |
| [data_ctrl_v2/](data_ctrl_v2/) | `train/`32本、`val/`8本の制御用Simulatorデータと生成ログ。主に出力・操作列を保存 | 今回の共通初期データ。生の実炉CSVではない |
| [data_multi_v2/](data_multi_v2/) | `large5/`・`medium6/`・`small4/`・`small5/`・`small7/`の複数プラント同定データと生成ログ。現在は各2本 | 別プラントの検討用。今回のA/Bには未使用 |
| [scripts/](scripts/) | データ生成、従来のAgent起動、モデル学習、制御、解析、作図、動画生成等を個別に実行する補助スクリプト | 一部の共通実装を参照。今回の主入口はこの中ではなく`research/run.py` |
| [figs/](figs/) | 従来の制御・解析結果JSON、目標値`setpoints.json`、PI設定`pi_gains.json`、モデル`cv_model.pkl`・`wm_bundle.pkl`等 | `setpoints.json`を固定参照へコピー。今回の結果の保存先ではない |
| [out/](out/) | ローカル動作確認と動画の生成物。`local_check/`は検査ログ・環境記録、`movie/`は既存制御GIF・preview・renderログ | 今回の結果は`research/runs/`に別保存 |
| [tests/](tests/) | 既存の物理・幾何・安定性・環境APIのテスト。`conftest.py`は共通テスト設定 | `research/tests/`の実験管理テストとは別 |
| `.venv/` | このMac用のPythonとインストール済み依存ライブラリ | 実験コマンドで使う環境。研究コード・学習データではない |
| `.pytest_cache/` | pytestが自動生成する検査のキャッシュ | 実験結果やテスト本体ではない |
| `.git/` | gitの履歴・ブランチ等の管理情報 | 実験データではない |

`__pycache__/`は各所にできるPythonの実行キャッシュ、`.DS_Store`はFinderの表示用情報です。
`src/wmf.egg-info/`はパッケージのインストール時にできる管理情報です。
これらをモデルや実験結果として読む必要はありません。

### `scripts/`を探すときの案内

| 用途 | ファイル |
|---|---|
| データ生成 | `gen_dataset.py`：空間場データ、`gen_control_data.py`：制御用データ、`gen_multiplant_data.py`：複数プラントの制御用データ |
| 従来のAgent実験起動 | `run_autoresearch.sh`：`integration/`向け、`run_control_research.sh`：`control_research/`向け |
| 制御モデル学習・閉ループ実行 | `train_cv_model.py`：4出力モデル、`run_control.py`：Simulatorを実機とみなす`Plant`と制御ループ |
| 制御条件の同定・解析 | `identify_pi.py`：PI係数、`pick_setpoints.py`：目標値選定、`analyze_pid.py`：PIDの解析、`measure_speed.py`：処理時間 |
| 幾何条件の比較 | `ab_geometry.py`：炉の幾何情報をモデルへ与える効果の比較。今回のA/B実験とは別 |
| 従来の実験履歴整理 | `extract_codex_history.py`・`codex_status.py`：履歴・進捗、`render_codex_table.py`：履歴表、`update_program_table.py`：課題文の操作応答表更新 |
| 個別の図 | `fig_control.py`：制御、`fig_deadtime.py`：遅れ、`fig_flow.py`：処理の関係、`fig_plants.py`：炉の形状、`fig_learning_rounds.py`：学習回数の比較、`fig_report.py`：レポート用図 |
| Demo素材・動画 | `make_demo.py`・`make_demo_figs.py`：予測Demo素材、`make_control_movie.py`：制御と温度場の動画 |
| まとめて実行 | `demo.sh`・`build_demo.sh`：既存Demo、`reproduce.sh`：制御データ等の再生成、`build_figs.sh`：既存図・レポートの生成 |
| 既存HTMLレポート | `build_report.py`：生成処理、`report_template.html`：テンプレート。今回の`research/report.py`とは別 |

この一覧は用途の案内です。各スクリプトには必要な入力・過去の設定・生成先があり、
全スクリプトが現在のcheckoutでそのまま実行済みという意味ではありません。

## 1. インストールと動作確認（3 分）

```bash
git clone <this repo> && cd WorldModel_fluid
python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q          # 物理の健全性（約 3 分、CPU）
```

GPU 機では `.venv/bin/pip install -U "jax[cuda12]"`。CPU でも動きます（開発は M シリーズ Mac）。

## 2. シミュレータを動かす

### 2a. 1 ステップずつ（モデルや制御器を差し込むならこれ）

```python
import numpy as np
from wmf.sim import IncineratorEnv

env = IncineratorEnv("configs/plants/plant_a.yaml", seed=0)
obs = env.reset()                       # obs["q"]: [64,64,7] 気相場, obs["bed"]: [64,4] 層
action = {                              # 制御周期 1 s ごとに与える
    "stoker_speed": 0.020,              # m/s        火格子速度   0.010–0.030
    "waste_feed":   1.30,               # kg/s       ごみ投入     1.0–1.6   (plant_a; 他は火格子長でスケール)
    "primary_air":  [1.25] * 5,         # Nm3/s/ゾーン 一次空気   1.0–1.5   (plant_a は 5 ゾーン)
    "secondary_air":[0.40] * 6,         # Nm3/s/ノズル 二次空気   0.2–0.6   (plant_a は 6 ノズル)
}
obs, reward, info = env.step(action)
info["t_exit"], info["o2_exit"], info["unburnt_bed"]   # 制御量（実機で測れるもの）
```

`obs["q"]` のチャンネルは `T, u, v, p, Y_F, Y_O2, Y_P`。ただし **制御器が使ってよいのは `info` の
測定値だけ**です — 実機は炉内の場を測れません。`scripts/run_control.py` の `Plant` クラスが
その制約を強制したラッパで、制御ループはこれを使っています。

### 2b. エピソードをファイルに（データ生成）

```bash
# 単一エピソード（ランダム操作、600 制御ステップ = 600 s）
.venv/bin/python -m wmf.sim.run --plant configs/plants/plant_a.yaml --actions random \
       --steps 600 --seed 0 --out out/ep_0000.npz

# 制御用の同定データ（制御量と操作量だけ。APRBS 励振で全操作範囲を網羅）
.venv/bin/python scripts/gen_control_data.py --frames 1200 --save-every 333 \
       --n-train 32 --n-val 8 --batch 4          # CPU で 1–2 時間
```

### 2c. プラントを変える

`configs/plants/plant_{a,b,c}.yaml` に炉のサイズ・傾斜・ゾーン数・ノズル数・煙道位置・
ごみ性状があります。`src/wmf/plants/generator.py` は 12 ファミリ（3 サイズ × 4 ゾーン数）から
プラントを手続き生成し、`mu.py` がその仕様を 12 次元ベクトル `mu_known` に正規化します。
格子は全プラント 64×64 固定なので、異なる幾何を `vmap` で同時に回せます。

## 3. World Model を差し込む・autoresearch で書かせる

`control_research/` が自己改善ループの作業場です（`karpathy/autoresearch` の構成）:

| ファイル | |
|---|---|
| `run.py` | **エントリーポイント。** `M` を学習 → `C` が `M` の中で計画して実機（シミュレータ）を制御 → `control_cost` を出力 |
| `program.md` | Agent への課題文と実行手順。読めばそのまま回せる |
| `worldmodel.py` | **Agent が編集する唯一のファイル。** アーキテクチャは指定していない（初期状態は線形状態空間モデル＝超えるべき床） |
| `prepare.py` | 同定データの読み込みと学習時間の予算（不変） |
| `evaluate.py` | 採点。**予測誤差ではなく制御性能**（目標追従の IAE + オーバーシュート + 操作量）で採点する（不変） |

```bash
cd control_research && ../.venv/bin/python run.py > run.log 2>&1
grep "^control_cost:" run.log                    # 低いほど良い。1 実験 ≈ 6–8 分（CPU）
```

`M` に要求されるインターフェースは 4 関数だけです（`program.md` 参照）:
`init_params / warm_up / predict / train`。中身は自由。

Codex で回すには `scripts/run_control_research.sh`（`--sandbox workspace-write`、専用ブランチ、
API キー不使用・ChatGPT サブスクリプション認証）。

## 4. 何を計算しているか（1 制御ステップ = 1 s = 333 積分ステップ）

**気相**（64×64、人工圧縮性 + Boussinesq 浮力 + 一段 Arrhenius）と
**1D ストーカ層**（乾燥 → 熱分解 → チャー燃焼、層温度 $T_b$ 付き）の結合。
物理の詳細・単位・値域は [INTERFACES.md](INTERFACES.md)。

時定数は実機のものです — ここが制御課題の核心です:

| | |
|---|---|
| 気相の応答 | 数秒（フリーボード滞留 2.4 s） |
| **火格子の滞留** | **400–1200 s**（実機ログ 660–720 s を挟む） |
| 投入・火格子速度 → 灰の未燃分 | 滞留時間ぶん遅れて現れる |
| 操作量のレート制約 | 各軸スパンの約 3 %/s（空気ダンパのフルストローク 30–60 s） |

実機の装備で入れているもの: 一次空気予熱器（450 K）、下部炉の耐火ライニング（1250 K）と
上部の水管壁（600 K）、二次空気ジェットの貫通（炉幅の 35 %）、炉全体を見る放射、層内の火炎伝播。
いくつかの集中定数（壁損失・放射・予熱）は運転点が実機並み
（炉温 1000–1300 K、O₂ 6–10 %、灰の未燃分 < 3 %）に乗るよう**校正した値**です。

## 5. 現在地と既知の限界

- **plant_a** は行動範囲の全隅で着火を保ち、NaN・速度クランプなし。同定データと制御ループはこれで回しています。
- 操作範囲はプラントごとに火格子長と発熱量でスケールします（`wmf.actions.envelope(cfg)`）。
  `IncineratorEnv.step` は範囲外の指令をその包絡線に射影します。
- 人工圧縮性のため圧力チャンネル `p` は定性的（最大 Mach ≈ 0.7）。温度・組成・速度は定量。
- 一段反応なので CO/NOx は存在せず、記録しているのは proxy（`reward.py`）。
- 実機ログ（2020-08）は解析済み（無駄時間 11–12 分、入力の条件数 5,320 = 運転データだけでは
  制御モデルを同定できない）。接続は未着手。

## 6. 実装本体 `src/wmf/` の見方

全体の配置は冒頭の「全体像」を参照してください。SimulatorやMPCの中身を読む場合は次の対応になります。

| 場所 | 役割 |
|---|---|
| `__init__.py` | `from wmf import Furnace`等で外へ公開する入口 |
| `furnace.py` | 操作を与えて炉を進める、簡単な呼び出し用ラッパ |
| `actions.py` | 操作範囲と、同定用のランダムな操作列（APRBS）の生成 |
| `reward.py` | rewardとCO/NOx等のproxy指標の計算 |
| `selfcheck.py` | インストール後の簡単なSimulator動作確認 |
| `sim/physics.py` | 気相の物理計算 |
| `sim/bed.py` | 火格子上のごみ層の物理計算 |
| `sim/rollout.py` | 気相・ごみ層を結合し、時系列や複数軌跡を計算 |
| `sim/env.py` | `reset()`・`step()`で使う環境API |
| `sim/run.py` / `sim/episode.py` | SimulatorのCLIとエピソード保存処理 |
| `plants/schema.py` / `plants/encode.py` | 炉の設定の読み込み・検査と、幾何・操作の固定格子への変換 |
| `plants/generator.py` / `plants/mu.py` | 複数の炉仕様の生成と、仕様を数値ベクトルへ変換する処理 |
| `control/mpc.py` | 学習モデル内で操作候補を比較し、次の操作を決めるMPC |
| `control/model.py` / `control/data.py` | 既存の制御用モデル・正規化とデータ処理 |
| `viz/render.py` | Simulatorの状態を画像として描画 |

各サブディレクトリの`__init__.py`は、Pythonパッケージとしての入口・公開名を定義します。
Agentが編集するモデル本体は実験ごとの`worldmodel.py`等なので、共通の`control/model.py`と区別してください。

## 7. 別マシン（NVIDIA GPU）で続けるとき

```bash
git clone https://github.com/Mi2ki510/WorldModel_plants && cd WorldModel_plants
make install-gpu          # venv + pip install -e ".[dev,figs,cuda12]"   （CPU なら make install）
make selfcheck            # 30 s。3 プラントで sim が動き、有限で、クランプが効かないことを確認
make test                 # 183 件（CPU で ~8 min、GPU ならもっと速い）
make baseline             # autoresearch の 1 実験（線形 M の床。CPU で control_cost = 1.819 だった）
```

- パスはすべてリポジトリ相対。`PY=python3 make ...` で別インタプリタも可。
- JAX は同じコードで GPU を使う。`JAX_PLATFORMS=cpu` を付けると強制 CPU。
- 一番簡単な入口は `from wmf import Furnace`（`src/wmf/furnace.py`）：入力を与えれば 1 秒ぶん進む。
  範囲外の入力は包絡線に射影、形の間違いは即エラー、状態は有限性を毎ステップ検査。
- **注意（正直に）**: `furnace.py` / `selfcheck.py` / `fig_learning_rounds.py` / `codex_status.py` は
  この Mac ではまだ実行していない（構文確認のみ）。最初に `make selfcheck` を回して、落ちたら
  そこから直してください。
- `control_research/worldmodel_codex_sep14c_unfinished.py` は Codex が 1 実験目の途中で
  止められたときの編集（線形モデルを大きく削って書き換えていた）。**未採点**なので
  `worldmodel.py` は床の線形モデルに戻してある。
- 学習 0/1/2 回目で目標に近づく図：`python scripts/fig_learning_rounds.py`（~20 min CPU）。
