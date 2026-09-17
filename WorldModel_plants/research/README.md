# 固定LLMでWorld Modelを開発する比較実験

今回のプロンプト・作業場所・読み書きの許可・ログの詳細は
[実験の動作解説](EXPERIMENT_WALKTHROUGH_20260915.md)を参照。

**初回の本実験は約64分で完了。** Aは最終評価で予測NRMSEを64.3%、平均制御costを16.7%改善した。
Bは予測が改善した一方で制御が悪化した。Bが生成条件を変更しなかったため、
診断によるデータ生成の優位性は未検証。[結果と限界・動画](RESULTS_20260915.md)を参照。

Codexが仮説とコード変更を提案し、外部runnerがSimulatorでのデータ生成、
World Modelの学習、予測・制御評価、採否判断を繰り返す。
LLM・参照Simulator・MPC・採点尺度を固定し、次の2条件を比較する。

| 条件 | 追加データ |
|---|---|
| A | 元のランダムAPRBS入力で生成 |
| B | Agentが誤差を分析し、操作範囲と長短の操作保持時間の混合を指定して生成 |

両条件ともモデルと学習pipelineを変更できる。4ラウンドずつ、各ラウンドに
1200制御ステップの軌跡を2本追加し、各候補を150秒の学習予算で比較する。
条件間で生成seedを揃え、AB→BAと実行順を交互にする。条件ごとの文脈は分離する。
同じ形式の診断情報・履歴を与えるが、他条件の結果は共有しない。

これは学習済みLLMを研究Agentとして使う実験。World Modelと学習用の実験条件を
改善する内側ループを実装している。Simulator本体の校正やLLM重みの再学習は次段階。

## 全体像：何がどこにあるか

`research/`は**実験を回す仕組みと、その実行記録をまとめる場所**です。
Simulator本体は親ディレクトリの`src/wmf/`、既存の制御実験は`control_research/`にあります。
リポジトリ全体の関係は[親のREADME](../README.md)を参照してください。

### 読む順番

1. このREADMEで構成と実行方法を確認する。
2. [RESULTS_20260915.md](RESULTS_20260915.md)で今回何が改善し、何が未検証かを読む。
3. [EXPERIMENT_WALKTHROUGH_20260915.md](EXPERIMENT_WALKTHROUGH_20260915.md)で実際のプロンプト、作業場所、権限、ログを確認する。
4. 実装を変更する場合は`Prompt.md`・`INTERFACES.md`、次に`run.py`と担当する処理のファイルを読む。

### 直下の資料・設定ファイル

| ファイル | 何のためのものか |
|---|---|
| [README.md](README.md) | この仕組みの入口。構成・実行コマンド・評価・保存先の案内 |
| [Prompt.md](Prompt.md) | **実験用Codexへの指示**。目的、手順、編集範囲、禁止事項。各runの作業コピーへ配布する |
| [INTERFACES.md](INTERFACES.md) | **今回の実験の実装契約**。モデルAPI、データの形、時刻対応、生成条件、採点。親の同名文書とは別 |
| [RESULTS_20260915.md](RESULTS_20260915.md) | 初回本実験の結果と解釈。採用モデル、改善・悪化、限界、動画・CSVへのリンク |
| [EXPERIMENT_WALKTHROUGH_20260915.md](EXPERIMENT_WALKTHROUGH_20260915.md) | 初回本実験の動作を詳しく説明。プロンプト、担当、作業ディレクトリ、読み書きの許可、ログの場所 |
| [VALIDATION.md](VALIDATION.md) | テスト・短縮実行・本実験の監査で何を確認したか |
| [.gitignore](.gitignore) | 実行結果の`runs/`をgit管理から除外する設定 |

### 直下のPythonファイル

| ファイル | 役割 | 主に確認・変更する場面 |
|---|---|---|
| [run.py](run.py) | **実験全体の入口・進行管理**。初期モデル、作業コピー、Codex起動、予算、採否、最終評価を管理 | 実験条件や進行手順を変える |
| [worker.py](worker.py) | runnerから渡された`job.json`を読み、生成・学習・評価のいずれか1処理を別プロセスで実行 | 数値処理への引き渡し、モデル保存、エラー処理を調べる |
| [data.py](data.py) | Simulator操作列と軌跡の生成、`.npz`読み込み、操作条件の被覆の集計 | 追加データの作り方・読み方を調べる |
| [evaluate.py](evaluate.py) | 連続予測のRMSE/NRMSE、固定MPCの制御指標、評価軌跡の保存 | 指標の定義・時刻対応を調べる。進行中の比較では固定 |
| [common.py](common.py) | JSON保存、ファイルhash、編集範囲検査、subprocessとtimeout、採点比率等の共通処理 | 実験の保護・失敗処理・共通の記録方法を調べる |
| [telemetry.py](telemetry.py) | CodexのCLI出力とsession JSONLを読み、応答別token・tool・モデル設定を取り出す | Agentの行動・使用量ログを解析する |
| [report.py](report.py) | 保存済み結果から比較CSV、時系列表、グラフ、GIF、Markdown/HTMLレポートを作る | 見せ方・集計を変える。学習や追加のSimulator実行は不要 |
| [__init__.py](__init__.py) | `research`をPythonパッケージとして扱うためのファイル | 通常は読む必要はない |

### 直下のディレクトリ

| ディレクトリ | 中身・役割 |
|---|---|
| [templates/](templates/) | 各runの共通初期モデルのひな形。`worldmodel.py`は線形状態空間モデルと学習、`pipeline.py`は学習を呼ぶ入口 |
| [tests/](tests/) | `test_research.py`に評価尺度、データ時刻、編集範囲、timeout、候補選択、ログ解析等のテスト。親の`tests/`は物理等のテスト |
| [runs/](runs/) | **1回の実行につき1ディレクトリ**。実際のコード・データ・モデル・ログ・結果を保存する |
| `__pycache__/` | Pythonが自動生成するキャッシュ。`templates/`・`tests/`にもできる。モデルやログではない |

`templates/worldmodel.py`は初期モデルで、Agentがそのファイルを直接編集するわけではありません。
Agentの編集先は`runs/<run_id>/workspaces/.../worldmodel.py`、採点に使う保存版は
`runs/<run_id>/experiments/<id>/source/worldmodel.py`です。

### コードとデータの流れ

```text
run.py                         全体を管理
  ├─ Prompt.md / INTERFACES.md / templates/ をrun内へコピー
  ├─ Codex                     workspaces/内でモデル・仮説・研究メモを編集
  ├─ common.py                 編集検査・時間制限・記録
  ├─ worker.py
  │    ├─ data.py              Simulatorから追加軌跡を生成
  │    ├─ 候補のpipeline.py    候補のworldmodel.pyを使って学習
  │    └─ evaluate.py          予測とMPC制御を採点
  ├─ telemetry.py              Codexの応答・tool・tokenを記録
  └─ report.py                 保存済みの結果・軌跡から図表と動画を作る
```

### `runs/`以下の全体像

現在保存されている主なrunは以下です。今後の実行でも新しいIDが追加されます。

| runディレクトリ | 内容 |
|---|---|
| [20260915T045134Z_09b058/](runs/20260915T045134Z_09b058/) | 短縮した動作確認。Codexを呼ばず、学習・制御を短縮。性能比較の根拠には使わない |
| [20260915T045740Z_d0631e/](runs/20260915T045740Z_d0631e/README.md) | 今回の本実験。A/B各4ラウンド、約64分、最終評価と動画まで完了。リンク先にrun内の全ファイル・ログ・配列の詳細案内 |

```text
runs/<run_id>/
├── reference/           そのrunの開始時に固定したコード・設定・初期データ
├── preflight/           Codexの事前確認。そのagent/に指示・起動・sessionログ
├── workspaces/
│   ├── A/round_01/ …    Aの各回でCodexへ渡したファイルと編集場所
│   └── B/round_01/ …    Bの各回。Aとは履歴を分離
├── experiments/
│   ├── baseline/       初期モデルの学習・検証
│   └── A_01/ 等
│       ├── source/     実際に採点する候補ソースと仮説・研究メモ
│       ├── agent/      Codexへの指示・stdout・stderr・session・token記録
│       ├── generation/ 追加データ、生成条件、生成ログ
│       ├── train/      学習済みmodel.pkl、学習結果、実行ログ
│       └── validation/ 予測・制御の検証結果、軌跡、実行ログ
├── final/
│   ├── data/           候補を選んだ後で新しく作った最終評価データ
│   └── baseline/ 等    選択済みモデルの最終評価と制御軌跡
├── report_process/     自動レポート生成プロセスの実行ログ
├── manifest.json       実験設定・実行状態・全体の時間
├── selection.json      最終評価前に固定した候補選択
├── final.json          最終評価の結果と成功判定
└── report.md / *.csv / *.png / *.gif 等
                        人間向けの集計・図・動画
```

`workspaces/`は**編集場所**、`experiments/`は**候補ごとの保存記録**、`reference/`は
**固定した土台**です。過去runの再現・調査には、現在の開発ファイルではなく、そのrunに保存したものを使います。
各JSON・CSVの役割は後半の「結果の保存先」、生ログの詳しい配置は[実験の動作解説](EXPERIMENT_WALKTHROUGH_20260915.md)に記載しています。

## 実行

このcheckoutに用意した `.venv`（Python 3.12、JAX CPU）と、ログイン済みの
Codex CLIを使う。既存の `control_research/` のファイルや `figs/wm_bundle.pkl`
を上書きせず、新しいrunディレクトリで実行する。

```bash
cd /Users/cls-lab/Git/NEDO_RSI/WorldModel_plants

# 数分の動作確認。Codexは呼ばず、学習5秒・短縮した制御評価を使う。
JAX_PLATFORMS=cpu .venv/bin/python -u research/run.py smoke

# 本実験。CodexによるA/B各4ラウンドと、未使用条件での最終評価まで実行。
JAX_PLATFORMS=cpu .venv/bin/python -u research/run.py compare
```

標準は `gpt-6-astra` / reasoning effort `low`。`--model` と `--effort` で
変更できるが、1回の比較内で切り替えない。モデル利用不可や認証エラー時は
別モデルへ黙って切り替えず、preflightのログを残して停止する。
ユーザーの通常のCLI設定は読み込まず、認証情報は既存のCodex認証を使う。

実験全体の上限は120分、開発ループは90分以内、各条件の開発消費時間も45分以内。
残りを最終評価とレポートに使う。学習は150秒をモデルへ渡し、終了待ちに15秒、
学習worker全体には起動等を含め185秒の上限を設ける。実際の時間・更新回数も記録する。
提案は1回300秒、生成は240秒、validationは240秒、最終制御評価は候補ごと600秒以内。
個々の上限よりrun全体の残り時間を優先し、timeoutでは子プロセス群も終了する。
終了記録と最小レポートの保存は、上限直後に数秒かかることがある。

開始時に表示されるrunディレクトリの`events.jsonl`に進捗が追記される。
別ターミナルで`tail -f research/runs/<run_id>/events.jsonl`を開くと、生成・学習・
評価・採否を確認できる。応答別tokenは各提案の終了時に保存され、最後に集計される。

予算不足なら両条件で試行を終えた共通ラウンドまでで候補を選ぶ。
失敗した試行も機会数に数え、失敗率を隠さない。最終評価の時間不足は欠損として残す。
途中停止後も保存済みの成果物は保持する。既存runへの上書き・自動resumeは行わない。

```bash
# 最大2ラウンドで終了する比較。採点方法は標準のまま。
JAX_PLATFORMS=cpu .venv/bin/python -u research/run.py compare --rounds 2 --minutes 60

# 保存済み結果からレポート・動画を再生成（Agentや学習は実行しない）。
.venv/bin/python research/run.py report --run-dir research/runs/<run_id>

# 既存のCodexチャット1件を解析する（原本は読み取りのみ）。
.venv/bin/python research/telemetry.py \
  /absolute/path/to/rollout.jsonl --out /absolute/path/to/analysis.json
```

## 別のCodexチャットに依頼する場合

```text
/Users/cls-lab/Git/NEDO_RSI/WorldModel_plants を作業先として、
research/README.md を読んでください。
準備済みの .venv で research/run.py compare を実行し、
A/B各4ラウンド、最大120分、最終評価・レポート・動画生成まで進めてください。
runnerが起動するCodexに実験案を提案させ、評価条件は途中で変更しないでください。
失敗やtimeoutも保存し、改善しなかった場合はその結果を報告してください。
```

## Agentの編集範囲と実験の保護

Agentが編集するのは各runの作業コピーだけ。`worldmodel.py`、`pipeline.py`、
Bの`sampling.json`、`hypothesis.json`、`ResearchState.md`を許可する。
ファイルの追加・削除・symlink・許可外変更は採用しない。
候補は必ず別ディレクトリへ保存し、最良の**学習済みbundleと対応するソース**を選ぶ。
git resetやcheckoutは使わず、既存の未コミット変更も保持する。

CLIのsandboxはworkspace-write。manifestと前後hashは実験条件の保護と監査用で、
任意のPythonコードに対する完全な情報隔離を保証するものではない。
Agentへは作業コピー外の閲覧・追加学習・Simulatorの直接実行を禁じる。
Agentはコード・実験条件を提案し、数値toolを実行するのはrunnerという役割分担。
Codexのtool利用とrunnerの処理はactorを分けてログに残す。

## データと評価

元の32学習エピソードと8検証エピソードをrun内へコピーする。候補の学習には
学習エピソードのみを渡す。追加データも学習用。生成に成功したデータは、
その回のモデルが不採用・学習失敗でも次の試行へ蓄積する。

`u[k]`を与えた結果が`y[k]`。未来操作が既知という条件の予測であり、
未知の未来操作を当てる予測ではない。1制御ステップは333×0.003=0.999秒。
出力は温度・O₂・ガス未燃分・灰未燃分。実測CO/NOx・蒸気・発電量は含まない。

- 予測：起点300、直前のWARMUP分だけを入力し、900ステップ先まで実測値で
  リセットせず予測。30・300・900の各区間について4出力のRMSEを計算する。
- 正規化：元の学習データの標準偏差を固定。4出力・3予測区間を等重みで平均する。
  データ追加によって分母が大きくなり、スコアだけが下がることを防ぐ。
  このNRMSEは運転範囲に対する百分率ではない。
- 制御：既存MPCを固定して400ステップの2シナリオを実行する。追従・overshoot・
  操作変化の元のcostを、固定尺度で計算する。MPC内部の正規化も共通に固定する。
  元のcostは`IAE_T + IAE_O2 + 0.7*IAE_bed + 0.5*overshoot + 0.1*move`。
  IAEは固定尺度で正規化した平均絶対追従誤差。ガス未燃分の追従重みは0だが、
  予測NRMSEには他の3出力と同じ重みで含める。全出力の改善を保証する採点ではない。
- 選択：`0.5*(予測NRMSE/初期NRMSE + 制御cost/初期cost)` が改善した候補を残す。
  片方が悪化した候補が選ばれる場合もあるため、最終成功条件は別に判定する。
- 最終評価：候補選択を保存してから新しい4軌跡を生成。制御は目標の変更方向を
  反転し、1600ステップ・切替300で評価。結果はAgentへ戻さない。
- 成功：最終評価の**予測NRMSEと制御costの両方**が初期モデルより小さいこと。
  BがAを両指標で上回ったかも別に判定する。

全試行で同じ学習seedを使い、モデル構造変更後も原則ゼロから学習する。
同じ秒数でも更新数・パラメータ数・FLOPsは一致しないので併記する。
JAXの同期と計時を追加した初期モデルで新しくbaselineを測るため、従来の
`control_research/run.py`で測った値との厳密な再現比較は行わない。

## 結果の保存先

`runs/<UTC日時とID>/`に以下を保存する。

- `report.md` / `report.html`：人間向けの比較表・限界・グラフ・動画リンク。
- `manifest.json` / `reference_hashes.json`：実行設定、環境、評価定義、固定ファイルhash。
- `environment.json`：Pythonと数値計算・可視化パッケージのバージョン。
- `reference/`：その実験で実際に使うSimulator・MPC・評価器・初期データのコピー。
- `workspaces/A/` / `workspaces/B/`：各回にCodexへ与えた指示・履歴・編集ファイル。
- `experiments/<id>/`：仮説、差分、ソース、生成データ、学習済みbundle、評価、失敗理由。
- `results.jsonl` / `comparison.csv`：各候補の評価・採否。失敗値はnull/空欄。
- `forecast.csv` / `control.csv`：予測区間・出力ごとの誤差と、制御シナリオごとの指標。
- `selection.json` / `final.json`：テスト前に凍結した候補選択と最終評価。
- `tokens.csv` / `responses.csv` / `timeline.csv` / `analysis.json`：行動・使用量・時間の解析。
- `resources.csv`：実験IDで結合できる学習更新数・パラメータ数・時間・データ数・操作被覆。
- `control_<candidate>.gif`：保存した最終評価の軌跡と温度場を再生。追加シミュレーションなし。

Agentのstdout JSONLとstderrは別ファイルに保存する。`thread.started`のIDに
一致する`~/.codex/sessions/`のJSONLだけをrun内へ保管し、実験IDを対応付ける。
集計対象はrunnerが起動したpreflightと各提案のCodex。実装・監視を行う親チャットや
別チャットのtokenは含めない。`tokens.csv`ではpreflightを別行として分離できる。
`token_usage_record`はresponse_idで重複を除く。累積token_countとは加算しない。
`timeline.csv`はproposal・tool・生成・学習・評価・採否を時刻順で結ぶ。
toolにresponse_idが直接ないログでは、tool呼び出し→usage→tool結果の順序から
対応を推定し、`response_link_method`へ明記する。対応が不明な場合は空欄にする。
後処理の再実行時には、そのコードhashも`analysis.json`へ保存する。
cacheは入力の内数、reasoning tokenは出力の内数。cacheを除いた入力と出力も
必要に応じて計算できるが、金額を推定するものではない。

CLIバージョンによって応答別usageがない場合は、`exec --json`のturn単位の
usageへ戻し、粒度を明記する。取得できないものを0 tokenの実験として扱わない。
API全再試行と正確な通信時間は未取得。toolの時刻差とAPI待ち時間を混同しない。
非公開の内部思考を復元・復号する処理はない。生ログはローカル保管し、
共有用の図表は必要な指標に絞って出力する。

失敗復帰率は「次回試行が存在する失敗のうち、次回が成功した割合」。
Tool selection accuracyは正解ラベルなしでは定義できないため自動計上しない。
操作条件の被覆は範囲、占有十分位、入力共分散の固有値として記録する。
これらの観測値は原因の証明ではなく、次の対照実験を設計する手掛かり。

## 検証と研究上の位置づけ

この環境でのチェック内容と短縮実行の記録は[VALIDATION.md](VALIDATION.md)を参照。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest research/tests tests -q
```

時刻対応と未来観測の遮断、固定尺度、元の制御costとの対応、操作範囲、
編集制限、timeout後の実行、token重複排除、部分ログ、共通ラウンドでの最良候補選択を確認する。

1回のA/B比較は探索的なDemo。統計的有意性、未知施設への汎化、実炉の制御、
スケーリング則はまだ主張しない。次は独立した複数回の比較と、追加軌跡数・
履歴管理方法・診断情報量を個別に変える実験へ広げる。

先行研究と今回の仮説は次を参照する。研究の狙いは「同じLLMでも、Simulatorで
何を試すかが、予測と制御の両方へ影響するか」を焼却炉環境で確かめること。

- [MLE-bench](https://arxiv.org/abs/2410.07095)：同じモデルでも実行基盤が結果に影響する先行例。
- [Do LLM Agents Have Regret?](https://arxiv.org/abs/2403.16843)：反復意思決定の評価。regret-lossの提案はモデル学習を伴う。
- [From Words to Actions](https://arxiv.org/abs/2405.19883)：環境を識別する探索の重要性。理論の前提を本実験へそのまま移さない。
- [From Debate to Equilibrium](https://arxiv.org/abs/2506.08292)：固定LLMの外部協調機構を改善する研究。今回の単一Agent実験とは分ける。
- [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)：CLIのJSON出力、model・sandbox指定。
