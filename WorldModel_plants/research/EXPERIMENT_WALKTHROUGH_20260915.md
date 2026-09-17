# 今回の実験はどう動いたか：プロンプト・作業場所・権限・ログ

対象runは`20260915T045740Z_d0631e`。以下は一般的なCodexの説明ではなく、
このrunに保存された指示・起動引数・ファイルmanifest・ログを確認した記録。
性能の結果は[RESULTS_20260915.md](RESULTS_20260915.md)にまとめている。
run内の各ディレクトリ、JSON・CSV・モデル・配列の役割は
[runのファイル案内](runs/20260915T045740Z_d0631e/README.md)を参照。

## 1. 誰が何を実行したか

今回の構成は、**Codexが実験案とコード変更を提案し、Python runnerが数値実験を実行する**ものだった。
Codexへ「好きなだけ学習・評価を実行して改善し続ける」という権限を渡した構成ではない。

| 担当 | 今回の役割 |
|---|---|
| この親チャットのCodex | `research/`の仕組みを実装・検証し、本実験を起動・監視。終了後にログ解析と説明資料を整備 |
| Python runner | 作業コピーの作成、Codexの起動、編集検査、生成・学習・評価の起動、予算管理、採否と最終候補の選択 |
| 実験用のCodex CLI | 指定された作業コピーで診断・履歴を読み、仮説とモデル等の変更を作る。1回の提案が終わると終了 |
| Python worker | Simulator軌跡の生成、World Modelの学習、固定MPCによる制御、固定の評価指標の計算 |
| 後処理 | 保存済みログ・予測・制御軌跡からCSV、図、GIF、レポートを生成 |

親チャットは`/Users/cls-lab/Git/NEDO_RSI`を作業ルートとし、実験起動コマンドは
`/Users/cls-lab/Git/NEDO_RSI/WorldModel_plants`で実行した。

```bash
JAX_PLATFORMS=cpu .venv/bin/python -u research/run.py compare
```

生成・学習・採点のworkerは、開始時にコピーして固定した
`runs/20260915T045740Z_d0631e/reference/research/worker.py`を使った。
このrunで使ったrunnerの実装も[reference/research/run.py](runs/20260915T045740Z_d0631e/reference/research/run.py)に保存されている。
現在の`research/`には実行開始後の記録・後処理の改善もあるので、当時の挙動を調べる場合は`reference/`を参照する。

## 2. Codexはどのプロンプトに従ったか

### 入口：各回の`agent/prompt.txt`

runnerは`codex exec`へ標準入力で短い指示を渡した。
Aの第1ラウンドに実際に渡した文字列は次のとおり。

```text
Read Prompt.md and follow it for condition A, round 1. Make exactly one experiment proposal in the editable files. The harness will generate data, train and evaluate after this turn. Use only this workspace. Do not run training or simulation yourself. Finish after writing hypothesis.json; report your hypothesis concisely.
```

「この作業場所のPrompt.mdを読み、Aの第1回として変更案を1つ作る。
生成・学習・評価はrunnerが行うので、自分では実行しない」という指示。
各回で条件名とラウンド番号が変わる。
原本は[experiments/A_01/agent/prompt.txt](runs/20260915T045740Z_d0631e/experiments/A_01/agent/prompt.txt)。

### 中心の指示：各作業コピーの`Prompt.md`

実験Agent用の目的・手順・制約をまとめた文書。
全回へ同じ凍結版をコピーした。
**今回実際に使った正本**は[reference/research/Prompt.md](runs/20260915T045740Z_d0631e/reference/research/Prompt.md)。
配布先の例は[workspaces/A/round_01/Prompt.md](runs/20260915T045740Z_d0631e/workspaces/A/round_01/Prompt.md)。

主な指示は以下だった。

- 将来予測と固定MPCによる制御の両方を改善する。
- `file_manifest.json`、`INTERFACES.md`、`feedback.json`、`ResearchState.md`を読む。
- 1回につき一貫した仮説を1つ作り、許可されたファイルのみ編集する。
- 構文チェックはよいが、Agent自身による学習・Simulator生成・評価は行わない。
- `hypothesis.json`へ仮説・変更・期待する効果を書き、研究メモを更新して終了する。
- Bは診断に基づいて次の生成条件を選ぶ。ただし、根拠があれば広い標準条件を維持してよい。
- 親ディレクトリ、他条件、過去の別実験、元checkout、Codex全体のsession、最終評価データを読まない。
- network、package追加、別LLM、subagent、git操作は使わない。
- Simulatorの物理式、MPC、採点方法、正規化尺度、データファイル、予算を変更しない。

Bが生成条件を変えなかったのは、変更権限がなかったからではない。
**標準条件を維持する判断も許したプロンプトで、実際にBがその判断を4回行った**。

### 指示を具体化する4つのファイル

| ファイル | Codexへ与えた情報 |
|---|---|
| `file_manifest.json` | 編集可能・読み取り専用のファイル名と作業範囲 |
| `INTERFACES.md` | 入出力の形、時刻対応、モデルAPI、生成条件の指定形式、評価方法 |
| `feedback.json` | 初期モデルと現在の最良モデルの検証指標、条件別誤差、同じ条件の過去試行の結果・仮説・学習情報 |
| `ResearchState.md` | Agentが前回までに書いた、証拠・却下した仮説・次の問いの要約 |

`feedback.json`には今回の条件、ラウンド、学習秒数、seed、最良候補IDも入っている。
失敗・不採用の試行は履歴に残るが、他条件の履歴と最終評価結果は渡していない。
例：[A第3回のfeedback.json](runs/20260915T045740Z_d0631e/workspaces/A/round_03/feedback.json)。

これらは実験用に追加した指示であり、Codexの全システム指示を`Prompt.md`だけで置き換えたという意味ではない。
CLI側の基本指示は保存sessionの`session_meta.payload.base_instructions`、
実行設定は`turn_context`にも記録されている。完全なAPI通信の保存とは区別する。

## 3. どのディレクトリで作業したか

以降の`RUN`は説明用の略記で、実際には次の絶対パス。

```text
/Users/cls-lab/Git/NEDO_RSI/WorldModel_plants/research/runs/20260915T045740Z_d0631e
```

```text
RUN/
├── reference/                   開始時に固定したSimulator・MPC・評価器・初期データ
├── preflight/agent/             CLIの事前確認（READYのみ）の記録
├── workspaces/
│   ├── A/
│   │   ├── round_01/            A第1回のCodexのcwd
│   │   ├── round_02/
│   │   ├── round_03/
│   │   └── round_04/
│   └── B/
│       ├── round_01/
│       ├── round_02/
│       ├── round_03/
│       └── round_04/
├── experiments/
│   ├── baseline/                初期モデルの学習・検証（提案Agentなし）
│   ├── A_01/ … A_04/            各提案のコード・データ・モデル・ログ・評価
│   └── B_01/ … B_04/
├── selection.json               最終評価前に固定した候補選択
├── final/
│   ├── data/                    最後に新しく生成した評価用4軌跡
│   ├── baseline/                初期モデルの最終評価
│   ├── A_03/                    Aで選んだモデルの最終評価
│   └── B_02/                    Bで選んだモデルの最終評価
└── report.md, 各種CSV, 図, GIF  集計と可視化
```

Codexへ`--cd RUN/workspaces/A/round_01`のように作業場所を指定した。
元の`control_research/worldmodel.py`を直接書き換えたわけではない。

各提案は**新しいCLI sessionで実行**した。1本のチャットを8回resumeした構成ではない。
前回から引き継ぐのは、最良候補のコードと、`ResearchState.md`・`feedback.json`による明示的な履歴。
A/Bの文脈は分離している。

実行順は`baseline → A_01 → B_01 → B_02 → A_02 → A_03 → B_03 → B_04 → A_04 → 最終評価`。
各ラウンドの順番をAB、BAと交互にした。数値実験は逐次実行している。

## 4. 読み取り・編集を許可したファイル

以下の9ファイルを各作業コピーに置いた。編集対象も内容を読むことは許可している。

| ファイル | Aの権限 | Bの権限 | 内容 |
|---|---|---|---|
| `worldmodel.py` | 読み書き | 読み書き | モデル構造、予測、損失、optimizer、学習関数 |
| `pipeline.py` | 読み書き | 読み書き | 学習データの使い方・学習手順の入口 |
| `sampling.json` | 読み取りのみ | 読み書き | 4操作の範囲と、長く保持する操作の確率 |
| `hypothesis.json` | 読み書き | 読み書き | `hypothesis`・`change`・`expected_effect`の3項目 |
| `ResearchState.md` | 読み書き | 読み書き | 次回へ渡す研究メモ |
| `Prompt.md` | 読み取りのみ | 読み取りのみ | 実験Agentの指示 |
| `INTERFACES.md` | 読み取りのみ | 読み取りのみ | 実装契約と評価方法 |
| `feedback.json` | 読み取りのみ | 読み取りのみ | runnerが渡す検証結果・履歴 |
| `file_manifest.json` | 読み取りのみ | 読み取りのみ | この権限一覧と作業範囲 |

実際の権限ファイル：[A第1回](runs/20260915T045740Z_d0631e/workspaces/A/round_01/file_manifest.json) /
[B第1回](runs/20260915T045740Z_d0631e/workspaces/B/round_01/file_manifest.json)。

Agent自身には生の`.npz`データを直接読ませる設計ではなかった。
**LLMが読むのは診断JSONとコード。学習コードが使う配列はworkerが学習時に渡す**。
workerが`pipeline.train(wm, data, stats, seconds, seed)`を呼ぶ際の`data`には学習用の
`y`・`u`配列が入り、検証・最終評価の配列は入れない。
実験用CLIが`reference/`のSimulator実装や評価器を自由に読むことも、今回の指示では禁止していた。

### 実際に変更したもの

全8提案で、hashが変わったのは`worldmodel.py`・`hypothesis.json`・`ResearchState.md`の3ファイル。
`pipeline.py`は編集可能だったが未変更。Bの`sampling.json`も編集可能だったが未変更だった。
`workspace_before.json`と`workspace_after.json`でこの違いを確認できる。

例えばA第1回のtoolログでは、`pwd`・`rg`・`cat`による場所とファイルの確認、
`apply_patch`によるモデル・研究メモ・仮説の編集が記録されている。
Agentは構文チェックも行える。学習・生成・採点のtool呼び出しはrunner側の記録に分けてある。

### 制限はどう実施したか

今回保存されたCLI起動設定は`--sandbox workspace-write`、モデル`gpt-6-astra`、推論`low`、
`web_search="disabled"`、`features.multi_agent=false`、`approval_policy="never"`。
さらにrunnerが編集前後のhashを比較し、許可外の変更やsymlinkを検査した。
`reference/`も各数値処理の前後にhashを検査した。

ただし、**読み取り範囲の禁止はプロンプト上の規則であり、manifestでOSの読み取り権限を閉じたわけではない**。
`workspace-write`という設定だけで、すべての外部ファイルの読取りや任意Pythonコードの動作が
完全に遮断されたと解釈しないこと。指示・編集検査・固定参照による実験管理と、
悪意あるコードも対象にした完全な隔離は区別している。

起動引数・cwd・開始終了時刻の原本は[experiments/A_01/agent/process.json](runs/20260915T045740Z_d0631e/experiments/A_01/agent/process.json)。

## 5. 1回の提案から次回まで

1. runnerが、その条件の最良候補から新しい作業コピーを作る。初回は共通の初期モデル。
2. runnerが同じ条件の履歴と診断を`feedback.json`へ書く。
3. Codexが指示を読み、モデル等と仮説・研究メモを編集して終了する。提案の上限は300秒。
4. runnerが編集範囲・仮説の必須項目・生成条件を検査し、`experiments/<id>/source/`へ保存する。
5. workerが1200ステップのSimulator軌跡を2本生成する。
6. workerが初期32本＋その条件の追加データで、150秒の予算でモデルを学習する。
7. workerが検証8軌跡の連続予測と、固定MPCによる2シナリオの制御を評価する。
8. runnerが初期値で割った予測NRMSEと制御costを等重みで平均し、最良値より下がれば採用する。
9. 結果を保存する。不採用でも生成済みデータと研究履歴を残し、次回は最良候補のコードから始める。

最終候補を決めた後に、新しい評価用4軌跡を生成し、1600ステップの制御シナリオで評価した。
最終評価を次のCodexの提案へ戻す処理はない。

## 6. 実行ログはどこにあるか

### まず見る集計ファイル

| 場所（RUNからの相対パス） | 何が分かるか |
|---|---|
| `report.md` / `report.html` | 全試行の採否、検証・最終評価、図、動画へのリンク |
| `events.jsonl` | runnerの進行。仮説、生成、学習、評価、採否などのイベント |
| `results.jsonl` | 初期モデルと各候補の結果。仮説、指標、学習情報、採否、エラー |
| `comparison.csv` | 各候補の評価・採否を比較する表 |
| `timeline.csv` | Codexのtoolとrunnerの処理を時刻順に並べた表。`actor`で担当を区別 |
| `responses.csv` | 応答ごとのsession・turn・response ID、入力・cache・出力token |
| `tokens.csv` | 提案ごとのtoken合計。事前確認は`preflight`行 |
| `resources.csv` | 学習秒数・更新数・パラメータ数・データ数・生成時間・操作被覆 |
| `forecast.csv` / `control.csv` | 予測区間・出力・制御シナリオ別の詳細な指標 |
| `analysis.json` | token・モデル設定・toolタイムライン等の集計、解析上の限界 |
| `manifest.json` / `environment.json` | 実験設定、全体時間、CLI・Python・パッケージのバージョン |

### Codex 1回分の生ログ

A第1回の場合のディレクトリは次。

```text
RUN/experiments/A_01/agent/
├── prompt.txt                  CLIへ渡した入口の指示
├── process.json                起動コマンド、cwd、開始終了、所要時間、終了コード
├── stdout.jsonl                codex exec --jsonの標準出力
├── stderr.log                  CLIの標準エラー
├── stream_events.jsonl         JSONイベントにrunner側の受信時刻を付けた記録
├── telemetry.json             sessionから抽出した応答usage、tool、モデル・turn情報
└── raw_sessions/*.jsonl        この提案に対応するCodex sessionのローカルコピー
```

B第2回なら`RUN/experiments/B_02/agent/`となる。
`stdout.jsonl`と`raw_sessions/*.jsonl`は同じ形式のファイルではない。
前者はCLIが出力した進行イベント、後者はCodexの保存session。
raw sessionには`session_meta`、`turn_context`、`response_item`、`token_usage_record`等が入っている。

元のsessionは`~/.codex/sessions/`にある。今回確認したA第1回の原本は以下。

```text
/Users/cls-lab/.codex/sessions/2026/09/15/
  rollout-2026-09-15T14-00-54-01a0a370-8009-73b2-977e-e9e44ea5ed68.jsonl
```

そのコピーは[experiments/A_01/agent/raw_sessions/](runs/20260915T045740Z_d0631e/experiments/A_01/agent/raw_sessions/)にある。
runnerはCLIの`thread.started`のIDと一致するsessionだけを保存した。
親チャットや他の無関係なチャットのログをまとめて取り込んではいない。

### 学習・Simulator・採点のログ

```text
RUN/experiments/A_01/
├── changes.diff                最良候補のソースから何を変えたか
├── source/                     実際に実行したモデル・pipeline・仮説・研究メモ
├── workspace_before.json       Agent編集前のhash
├── workspace_after.json        Agent編集後のhash
├── source_hashes.json          保存した候補ソースのhash
├── training_data.json          学習に使ったファイルパスとhash
├── generation/
│   ├── job.json                workerへ渡した生成条件・seed
│   ├── result.json             各軌跡の保存先・長さ・生成時間
│   ├── episodes/*.npz          生成した操作・出力・metadata
│   └── process/                stdout.jsonl, stderr.log, process.json等
├── train/
│   ├── job.json                データ・ソース・学習予算等
│   ├── result.json             実際の学習時間・更新数・パラメータ数等
│   ├── model.pkl               学習済みパラメータ・正規化情報等
│   └── process/                workerの実行ログ
├── validation/
│   ├── job.json / result.json  検証条件・指標
│   ├── forecast.npz            予測と正解
│   ├── control_temp_step.npz   温度シナリオの操作・出力・目標
│   ├── control_o2_step.npz     O₂シナリオの操作・出力・目標
│   └── process/                workerの実行ログ
└── experiment.json             この提案の仮説・結果・採否をまとめた記録
```

数値workerのログは`train/process/stderr.log`のように**もう1段`process/`が入る**。
Agentのログは`agent/stderr.log`であり、場所が異なる。
数値workerの`stdout.jsonl`は生成結果等で、Codexの会話ログではない。

異常があればworker側に`error.json`、候補側に`exception.txt`、run全体に`run_error.json`等が保存される。
今回の本実験に異常終了・timeoutはなかった。
最終評価の生データと実行ログは`RUN/final/baseline/`・`RUN/final/A_03/`・`RUN/final/B_02/`にある。

## 7. どこまで解析できるか

今回記録した48応答・39 tool呼び出しは、事前確認1回と実験提案8回の分。
入力928,181 token（うちcache721,536）、出力13,674 tokenを取得した。
実装・監視をした親チャットのtokenはこの集計には含まれない。

- 応答単位のusageはresponse IDで重複を除いた。累積token記録をさらに足していない。
- 入力のうちcache分、出力のうちreasoning分は内数なので二重加算しない。
- session・turn・response・toolを実験IDへ結び付けた。
- toolのresponse IDは生ログに直接ない場合があり、イベント順からの推定と明示している。
  この追加対応は、今回の後処理では`timeline.csv`・`analysis.json`側で確認できる。
- toolの所要時間、CLI全体の所要時間、数値workerの所要時間は区別して保存している。
- APIの全HTTP通信・全再試行・正確な通信時間を保存したものではない。
  非公開の内部思考を完全に復元する解析も行っていない。

## 8. 手元で見るためのコマンド

以下は保存済みファイルを見る例。本実験を再実行する必要はない。

```bash
cd /Users/cls-lab/Git/NEDO_RSI/WorldModel_plants

# Agentが実際に従った中心の指示
cat research/runs/20260915T045740Z_d0631e/workspaces/A/round_01/Prompt.md

# その回に与えた診断と履歴
cat research/runs/20260915T045740Z_d0631e/workspaces/A/round_03/feedback.json

# モデルの変更差分
cat research/runs/20260915T045740Z_d0631e/experiments/A_03/changes.diff

# 実験の進行ログ。別runを実行中ならそのrun IDに置き換えて使う
tail -n 30 research/runs/20260915T045740Z_d0631e/events.jsonl

# ブラウザで図を含むレポートを開く（Mac）
open research/runs/20260915T045740Z_d0631e/report.html
```

`runs/`はgitの対象外。実行結果を別PCや別checkoutへ持っていく場合は、このrunディレクトリをコピーする。
今回の実行時のチェックは[development_audit.json](runs/20260915T045740Z_d0631e/development_audit.json)と
[final_audit.json](runs/20260915T045740Z_d0631e/final_audit.json)に保存されている。
