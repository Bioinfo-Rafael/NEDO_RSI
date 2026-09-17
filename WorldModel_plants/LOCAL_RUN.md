# このMacでの実行

clone先は`/Users/cls-lab/Git/NEDO_RSI/WorldModel_plants`です。
取得したコミットは`b5c471dda8f12d3c2451d2da642521db178d988c`。

Apple M4 / メモリ24 GB / macOS 26.5で、Python 3.12.14の独立した`.venv`を作成し、`pip install -e '.[dev,figs]'`でCPU向け依存関係を導入しました。JAX 0.11.1は`CpuDevice(id=0)`を認識しています。インストール済みバージョン一覧は`out/local_check/environment.txt`です。

## シミュレータの自己診断

```bash
cd /Users/cls-lab/Git/NEDO_RSI/WorldModel_plants
JAX_PLATFORMS=cpu .venv/bin/python -m wmf.selfcheck
```

3種類の炉で入力形状の検査、操作範囲への補正、状態が有限値であることなどを確認します。JAXの初回コンパイル時間もかかります。

## テスト

```bash
cd /Users/cls-lab/Git/NEDO_RSI/WorldModel_plants
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/ -q
```

## World Modelの学習から制御評価まで1回実行

```bash
cd /Users/cls-lab/Git/NEDO_RSI/WorldModel_plants/control_research
JAX_PLATFORMS=cpu ../.venv/bin/python -u run.py > run.log 2>&1
tail -n 15 run.log
```

付属の`data_ctrl_v2/`からWorld Modelを学習し、固定のMPC制御器がそのモデルで操作を計画します。焼却炉シミュレータで2つのシナリオを各400制御ステップ評価し、最後に`control_cost`と学習・評価時間を出力します。小さいcontrol_costほどよい評価です。学習の標準予算は150秒ですが、初期化や制御評価時間は別です。

`figs/wm_bundle.pkl`に学習済みモデルを保存します。同じコマンドを再実行するとこのファイルは置き換わるため、残したいモデルは先に別名で保存してください。

これは1回の学習・評価です。Codexにコード改良を繰り返させる際は`control_research/program.md`が別の指示書になります。

## 今回の動作確認記録

2026-09-14に、このMacのCPUで確認しました。自己診断はローカル修正後に3プラントすべて成功し、全体で約7秒でした。元の183件と追加3件を合わせた全186テストは成功し、pytestの実行時間は60.82秒でした。

標準設定（学習予算150秒、seed=0）のWorld Model学習・MPC制御評価も終了コード0で成功しました。

| 項目 | 実測結果 |
| --- | ---: |
| 自己診断全体 | 6.951秒 |
| テスト全体（起動込み） | 61.304秒 |
| 学習・制御評価全体（起動込み） | 186.006秒 |
| 学習時間（スクリプト報告） | 150.5秒 |
| 制御評価時間（スクリプト報告） | 35.0秒 |
| control_cost | 1.842310 |
| temp_step_tracking | 1.377816 |
| o2_step_tracking | 1.571170 |
| アンサンブルのモデル数 | 5 |

自己診断・テストは修正後のオンラインAPIも確認しています。標準制御評価は元から独自のコンパイル済み`Plant`ラッパを使うため、今回の`env.py`の高速化はそのスコア計算経路を変更しません。

学習は時間予算で打ち切るため、同じseedでもマシン負荷やライブラリ・学習更新回数によってスコアが変わります。上記は今回の実行結果であり、リポジトリのREADMEにある他環境のスコアとの厳密な再現比較ではありません。

### 起動確認中に行ったローカル修正

元の`src/wmf/sim/env.py`は、1制御ステップ内の333積分ステップをPythonループで実行していました。初回の自己診断は約79秒経過しても1プラント目を完了しなかったため停止し、オンラインAPIから既存のJITコンパイル済み`rollout()`を1フレーム分呼ぶよう変更しました。物理式、制御器、World Model、評価条件は変更していません。

追加した`tests/test_env.py`では、3プラントそれぞれについて、元のPythonループを独立に計算し、変更後の状態・診断値・報酬が数値許容誤差内で一致することを確認しました。3件とも成功しています。長時間・多数積分ステップにおけるビット単位の一致を保証するテストではありません。

変更はローカルの未コミット差分です。初回の中断ログは`out/local_check/initial_interrupted_*`に残しています。

実行ログと終了コード・計測時間は`out/local_check/`に保存します。

- `selfcheck.log`：3プラントの自己診断
- `tests.log`：付属テスト
- `baseline.log`：学習とMPC制御評価
- `checks.json`：各実行の終了状態と実時間

## シミュレーションを動画で見る

作成済みの[制御シミュレーションGIF](out/movie/control_movie.gif)で、400秒分の挙動を約16秒で再生できます（200フレーム、約7.4 MB）。[最後のフレーム](out/movie/preview.png)も保存しています。

左はシミュレータの炉内温度分布（K）、右は出口温度・出口O2・灰の未燃分です。赤い破線が目標、青線がシミュレータの値。最初に履歴を蓄積してからMPCが操作し、150秒付近で目標を変更します。物理シミュレータの温度分布であり、実炉の撮影映像ではありません。初期モデルの制御結果なので、目標からのずれやオーバーシュートも表示されます。

既存の`scripts/make_control_movie.py`で、学習済みの`figs/wm_bundle.pkl`を使って生成しました。動画ファイル自体はclone時には含まれていませんでした。

```bash
cd /Users/cls-lab/Git/NEDO_RSI/WorldModel_plants
mkdir -p out/movie
JAX_PLATFORMS=cpu .venv/bin/python scripts/make_control_movie.py \
  --steps 400 --model figs/wm_bundle.pkl --out out/movie/control_movie.gif
```

別の可視化として`python -m wmf.viz.render <episode.npz> --out out`もあります。こちらは保存済みの場データから温度・酸素などのGIFを作る入口で、今回作成したものは上記のMPC制御動画です。

## 手元のCSVとの関係

このレポの標準実行はシミュレータと付属の同定データを使います。親フォルダの`200801-0831 (1).csv`は読み込みません。README上でも実機ログへの接続は未着手とされています。CO/NOxも詳細な化学種として実装されているわけではなく、proxy（代替指標）です。

シミュレータが実行できることと、実機の予測・制御精度が検証されていることは別です。複数指標予測・policy研究のための実験基盤として利用できます。
