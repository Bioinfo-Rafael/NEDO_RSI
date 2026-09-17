# Meta Search RSI — Phase 1

**過去の実験・探索履歴を使い、「次にどの枝を探索するか」という戦略を比較・改善するオフライン実験基盤です。** Codexを候補policyの提案役として接続できます。

- **Purpose**: 過去経験の再利用と、探索budgetを配分するpolicy programの改善。
- **Input DB**: 既存の `experience_store/experience.db` と `public_experience/*.db`。すべて読み取り専用。
- **Output**: `outputs/` のmemory、replay履歴、評価結果、採用policy。
- **Main entrypoint**: `python3 -m meta_search_rsi.cli ...`（`PYTHONPATH=src`）。Codex loopは `./autoresearch.sh`。
- **Review Guide**: [REVIEW_GUIDE.md](REVIEW_GUIDE.md)。構造は [ARCHITECTURE.md](ARCHITECTURE.md)、出典は [PAPER_MAP.md](PAPER_MAP.md) / [PROVENANCE.md](PROVENANCE.md)。

## 何を作ったのか

最終目標の自律研究Agentに向けて、まず「過去の履歴から、探索の進め方を検証する部分」を作っています。新しいモデルの学習や、研究コードの自動改変・実験実行はまだ担当しません。

```text
既存experience.db
    ↓ 読み取り専用
過去の探索木 ──→ Atom / Skill ──→ 関連経験の検索
    │                                  │
    └─────────→ Meta Search Policy ←───┘
                         ↓ 探索先を選ぶ
                 Historical Replay
                         ↓ 固定評価のfeedback
                Codexが候補policyを提案
                         ↓ 改善時だけ採用
                  次のpolicy評価へ
```

### ① 過去の探索木を読む

`runs` と `experiences.parent_id` から親子関係を復元します。一本道（linear）・分岐のある木・欠落を含む森を扱います。元データにないedgeは追加しません。

```text
Baseline
├─ 実験A
│  └─ 実験Aの改良
└─ 実験B
```

### ② 過去経験をmemoryとして整理する

| 種類 | 単位 | 内容 |
|---|---|---|
| Atom | 親→子の1遷移 | どんな状態で、何を行い、どうなったか |
| Skill | 複数stepの経路 | どんな順序で行動したか |

AtomはState / Goal / Action / Outcomeを保存し、Skillは実際の行動列をまとめます。現時点の抽象化はLLMなしの簡易版で、原文を保持し、指定entityだけを置換します。研究の本質を理解して高度な教訓を自動生成する機能は未実装です。BM25による単語検索で関連経験を取り出し、元DB/run/nodeの出典も返します。

### ③ 次にどの枝へbudgetを使うか決める

Meta Policyは `DEEPEN`（枝を伸ばす）、`WIDEN`（別枝を開く）、`REVISIT`（以前の枝へ戻る）、`STOP` を選びます。「どんなNNを作るか」「どんなSimulator式に変えるか」は将来のTask Agentの担当です。

### ④ 保存済みの結果で戦略を比較する

replay環境は過去のtree全体を持ちますが、policyには最初はrootだけを見せます。policyがtargetを選ぶと、その先の保存済みchild/resultを1つ公開します。これにより「同じ履歴でも探索順を変えると、限られた予算でどこまで到達できたか」を比較できます。実験は再実行せず、未記録のaction/resultも予測しません。

### Codexが担当する部分

**Codexはpolicyを改善する開発者役、Python側は固定評価器です。** Codexは仮説と候補コードをJSONで返し、Python側が構文・観測境界を検証してreplayし、改善時だけbestを更新します。memory生成・検索・replay・採点は引き続きAPI不要です。自動提案を有効にするときだけCodexの認証・ネットワーク・利用枠を使います。

| 現在あるもの | Phase 2以降 |
|---|---|
| 過去履歴の読み込み・検索 | 新しい研究実験の実行 |
| 保存済みtreeでの戦略比較 | 未知actionの結果予測 |
| Codex提案→固定評価→採否の有限ループ | Simulator / World Modelの自動改良 |
| CLIとJSON出力 | GUI・GoT Viewer連携 |

## まず動かす

Python 3.11以上、標準ライブラリだけで動きます。今回の確認環境はPython 3.14.5。API key、Codex、GPU、追加Python packageは不要です。

```bash
cd /Users/cls-lab/Git/NEDO_RSI/meta_search_rsi
./examples/demo.sh
```

この1本で、11nodeの実tree表示、既存local 3 runからのatom/skill生成、検索、replay、6 baseline＋candidateの比較を実行します。評価はOpenEvolve 5 runとSWE-smithのlinear 1 run、各8要求。実際の研究コード・NN・外部APIは実行しません。再実行ごとに別studyを作るため、過去trialを上書きしません。

依存するDBの場所とrun IDは `examples/demo.sh` に明記しています。DBファイルはこの新directoryに同梱しません。別PCでは手元にあるrunを下記CLIで指定してください。

## 実装したもの

| Component | File | 入出力・役割 |
|---|---|---|
| read-only loader | [db.py](src/meta_search_rsi/db.py) | 既存 `runs` / `experiences` をSQLで読む。`raw_events` は今回未ロード |
| tree / forest / prefix | [tree.py](src/meta_search_rsi/tree.py) | 実 `parent_id` のみを使用。cycle/duplicateを拒否。欠落parentを保持 |
| SGA-style atom | [atoms.py](src/meta_search_rsi/atoms.py) | 実parent→childをState/Goal/Action/Outcomeへ写像 |
| structural skill | [skills.py](src/meta_search_rsi/skills.py) | 実root→leaf経路をworkflow化し、同じ抽象step列だけをまとめる |
| evidence retrieval | [retrieval.py](src/meta_search_rsi/retrieval.py) | metadata filter → BM25 → 成功/失敗/別構造の証拠 |
| Meta Search Policy | [policy.py](src/meta_search_rsi/policy.py) | DEEPEN / WIDEN / REVISIT / STOP。具体的なtask actionは生成しない |
| historical replay | [replay.py](src/meta_search_rsi/replay.py) | root-only開始、選択後に実childを1つ公開 |
| policy改善 | [autoresearch/](autoresearch/) | 固定評価器とmemory、限定Python candidate、keep/discard |

LLM抽象化・embedding・rerankerは実装していません。memory/replay本体はLLMなしのfallbackです。Codexによるpolicy提案だけ任意で有効にできます。Simulator/World Modelの生成・予測・co-evolution・NN policyの学習もPhase 2以降です。

## 個別CLI

以下はこのdirectoryで実行します。`PYTHONPATH` は各コマンド内だけで指定し、shell設定ファイルは変更しません。

### runを探す・木を見る

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m meta_search_rsi.cli runs --pattern 'openevolve::*'
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m meta_search_rsi.cli tree \
  --run-id openevolve::MyOpenEvolve::21d972d0-0b7d-49dd-8f88-f98a26dbc6ee
```

terminalに親子関係をインデント表示し、`outputs/tree.json` に全選択runを保存します。GoT Viewerとは独立しています。GUIは追加していません。

### memoryを作る

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m meta_search_rsi.cli build-memory \
  --db ../experience_store/experience.db \
  --run-id run_4a1b8ad7c89c78d032bf0310 \
  --run-id run_56709ea154983586c368ff29 \
  --run-id run_660e45cf1cdc2d1538b777eb
```

`outputs/memory/memory.json` にatomとskillを保存します。`.db` が入力、memory JSONは派生生成物です。原DBを更新したり新しいDB/schemaを作ったりしません。全DBの暗黙ロードを避け、`--run-id` を列挙します。

- atom: `raw_state`, `raw_goal`, `raw_action`, `raw_outcome` と `abstract_*` を保持。goalは明示された場合のみ。失敗も記録。
- de-lexicalization: `configs/default.yaml` の `entity_slots` で明示したentityだけをtyped slotへ置換。例えば `{"forecast_loss":"<OBJECTIVE>"}`。数値の0.2→0.1は維持し、意味を推測して「補助目的」と言い換えない。
- skill: `skill_id`, `name`, `preconditions`, `workflow_steps`, `success_pattern`, `failure_pattern`, `source_run_ids`, `source_node_ids` を持つ。欠落actionはNULLのまま。patternは元の結果の証拠であり、LLMが蒸留した一般法則ではない。
- 出典: `source` は入力DBの絶対path、`source_run` / `source_nodes` は元のID。元DBでnodeからraw_eventsへ追跡できる。

### retrieve

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m meta_search_rsi.cli retrieve \
  --query 'evaluate model optimization failure loss' --structure tree
```

`outputs/memory/retrieval.json` にmemory ID、source、source run/node、relevance、raw/abstract textを返します。`--kind atom` / `--kind skill`、`--source` で絞れます。`--run-id` は検索から除外するrunです。`--structure` は別構造の証拠を選ぶための現在構造で、硬いfilterではありません。該当する失敗・別構造の証拠がなければ生成しません。

### replay・比較

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m meta_search_rsi.cli replay \
  --run-id openevolve::MyOpenEvolve::21d972d0-0b7d-49dd-8f88-f98a26dbc6ee \
  --policy meta_memory --budget 8
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m meta_search_rsi.cli evaluate \
  --run-id openevolve::MyOpenEvolve::21d972d0-0b7d-49dd-8f88-f98a26dbc6ee \
  --study outputs/autoresearch/my-study --label v1
```

policy: `random`, `dfs`, `bfs`, `current_best`, `meta_free`, `meta_memory`。評価時はこれら6つと現在のcandidateを比較します。

forestの場合は単独replayの `--root-id <既存component root ID>` で対象を選べます。既定は記録順の最初のrootです。**複数rootを人工的に接続せず、選択componentだけを評価**します。`component_nodes` と `excluded_component_nodes` を結果に記録します。

## prefix-onlyとは

開始時にpolicyが見るのは選んだrootだけです。各stepで:

1. 観測済みnode・その中の子数・短い結果・改善履歴・memory・残budgetを渡す。
2. policyが観測済みtargetを選ぶ。
3. replay環境がそのtargetの次の実childを公開する。child順は元の `sequence_index, node_id`。
4. childがなければ空probeとして記録し、そのtargetの枯渇を知る。

空probeも1budgetです。未観測childの有無で事前に候補を絞りません。policyには元node ID・未観測score・run全体の集計を渡しません。node IDは観測順の `P0000` 等に置換します。memoryも対象runと同task_nameからのものを除外します。

`DEEPEN/WIDEN/REVISIT` は探索配分の意図を記録するlabelです。historical環境の遷移は全て「選択した親から次の実childを返す」で、labelごとに未記録のactionや結果を作ることはありません。

## scoreの読み方

- `raw_best`: sourceの数値をそのまま保持。異taskのraw scoreを平均しない。
- `rank_attainment`: 選択component内のdistinct score順位を0〜1に正規化。minimizeも考慮する。
- `normalized_attainment`: 同componentの最悪値〜最良値を0〜1に線形変換した別指標。
- `rank_auc`: 各要求後のrankの平均。STOPした場合は最終値で残budget分を埋める。
- `objective`: `rank_attainment - 0.01 × 使用要求数 / budget`（係数は固定config）。

score名・方向が不明、またはcohortが複数あるrunではscalar metricをNULLにし、集計から除外します。raw_bestはcohort別に保持します。全scoreが1値なら、観測後のrankは1で、選択効果を測れない例です。最初からrootが最良の履歴もあり、高rankだけで探索効果を主張できません。

## Codexを使ったautoresearchへの統合

### 1. 新しい固定studyを準備する（モデル呼出なし）

最初に上記 `build-memory` を実行します。次の例は公開2 runを固定し、初期policyを評価してpromptを作るだけです。

```bash
./autoresearch.sh --study outputs/autoresearch/codex-study \
  --run-id openevolve::MyOpenEvolve::710ca863-c3ea-4d3a-88bb-e3444cec5c4a \
  --run-id openevolve::MyOpenEvolve::8b752b40-098d-4f6f-a2a1-1eed187e3cf2 \
  --dry-run
```

study内の `codex/0001/prompt.md` と `schema.json` で、Codexへ渡す内容を確認できます。promptには採用済みpolicy、許可構文、集約feedbackを入れ、DB全体・raw tree・secretは埋め込みません。評価中のcandidateが見るのは観測prefixのみですが、改善役がdevelopment評価feedbackを見ることは意図した設計です。

### 2-A. 現在のCodex sessionで使う

Codexに次のように依頼します。

```text
meta_search_rsi を作業directoryにし、README.md と autoresearch/program.md を読んでください。
outputs/autoresearch/codex-study/codex/0001/prompt.md を使い、候補を1つ作ってください。
{"hypothesis":"仮説", "code":"# ORIGIN: ORIGINAL\n..."} のJSONとして
outputs/manual-proposal.jsonへ保存してください。
./autoresearch.sh --study outputs/autoresearch/codex-study --iterations 1 \
  --proposal-file outputs/manual-proposal.json で評価し、採否と根拠を報告してください。
固定評価器、config、DB、memory、manifestは変更しないでください。
```

この方法では別のCodexプロセスを起動しません。追加のAPI keyも不要です。既存のCodexベースautoresearchに接続する場合も、**候補生成側がこの2-field JSONを書き、評価側が `--proposal-file` を呼ぶ**だけです。

### 2-B. Codex CLIで有限回の自動ループを回す

専用認証は `outputs/codex-home` を使います。既存 `~/.codex` のauth/configをコピー・変更しません。未ログインなら一度、利用者自身が次を実行します。

```bash
mkdir -p outputs/codex-home
env CODEX_HOME="$PWD/outputs/codex-home" codex \
  -c 'cli_auth_credentials_store="file"' login
```

ChatGPTでログイン後:

```bash
./autoresearch.sh --study outputs/autoresearch/codex-study --iterations 3
```

任意で `--model <利用可能なモデル名>`、`--codex-timeout 300` を指定できます。最大100候補/呼出、既定3候補です。API keyによる別provider呼出は追加していません。Codex呼出は `src/meta_search_rsi/llm.py` に集約し、read-only sandbox、ephemeral session、JSON schemaを指定します。authは専用homeのfile保存に限定し、ChatGPT loginを指定します。

仕様根拠: [OpenAI公式 non-interactive mode](https://developers.openai.com/codex/noninteractive)、[認証とCODEX_HOME](https://developers.openai.com/codex/auth)、手元の `codex-cli 0.154.0` の `exec --help`。新しいバージョンのCLI動作は別途確認してください。

### 採否・失敗・再開

- 初回はseed policyを評価し、比較の基準を作る。数値objectiveが得られないstudyはCodexを呼ぶ前に停止。
- 候補はstudy内へ保存して限定ASTで評価。`keep` のときだけbestを更新する。`discard` / `crash` ではbestを保持。
- 不正JSONは `invalid_proposal`。認証失敗・CLI異常終了・timeoutは `codex_error` として記録し、そのループを停止。
- 同じcommandを再実行すると保存済みbestとfeedbackから継続し、新しい番号を使う。未完了directoryも上書きしない。中断した候補を自動的に採用しない。
- 同じstudyでの二重loop・同時評価はlockで拒否。Ctrl-Cとtimeout時は起動したCodex process groupを停止する。
- `autoresearch/policy_candidate.py` は初期seedとして保持し、自動ループが書き換えるのはstudy内候補だけ。採用版の正本はtrial snapshot、便利用のcopyが `best_policy.py`。
- 原DB、固定file、memory/config/input snapshotの検証は評価前後に実行する。基盤更新後は新studyを作る。古いmanifestを書き換えない。

| 保存先（study内） | 内容 |
|---|---|
| `best.json`, `best_policy.py` | 現在の採用版とobjective |
| `trials/<label>/result.json` | 固定baselineと候補の評価、crash理由 |
| `results.jsonl` | 候補hash、仮説、objective、keep/discard/crash |
| `codex/<番号>/prompt.md`, `proposal.json` | 提案役への入力と構造化出力 |
| `codex/<番号>/events.jsonl`, `stderr.log` | 実Codex CLIのログ（手動/dry-runでは作らない） |
| `codex/<番号>/iteration.json`, `codex_iterations.jsonl` | 呼出引数、提案・実行失敗、再開履歴 |
| `cases.json`, `memory.json`, `config.json`, `manifest.json` | 固定入力とhash |

[AGENTS.md](AGENTS.md) にCodex向けの作業規則、[autoresearch/program.md](autoresearch/program.md) にpolicyの入出力契約を記載しています。実モデル連携は利用者の認証と利用可能枠に依存します。今回の開発検証は模擬CLI・手動JSON・dry-runで行い、実モデル生成は実行していません。

## 検証

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

28 tests（Codex連携テストを含む）: tree復元、cycle/duplicate拒否、forestのroot-only、親→子atom、score方向/異metric、skill/retrieval出典、memory除外、hidden score/ID/子数の非漏洩、6 baselineの決定性、network禁止、policy検証、限定candidate、read-only DB、固定評価器/入力改変拒否。

### Codex接続のsmoke test

`outputs/autoresearch/codex-integration-20260917` で公開OpenEvolve 2 runを使い、dry-runによるprompt生成と、手動JSONによる評価を確認しました。同一policyを候補として渡した結果、objectiveは `0.7292857143` のままで `discard` となり、採用済みbestを保持しました。これは接続確認であり、探索性能の改善を示す結果ではありません。

28件の自動テストでは模擬CLIを使ったsubprocess接続、keep/discard/crash、異常終了、timeout、再開も確認しています。**実Codexモデルによる候補生成は未実行**です。最新studyは `outputs/final_study.txt`、接続ログは `outputs/codex-integration-dry-run.log` / `outputs/codex-integration-manual.log` に保存しています。

## 論文との関係・限界

Dream-RSI、SGA-MCTS、LifeMemのconceptを使う新規実装です。著者コードのコピー・importはありません。WorldEvolverは公式コードも調査したうえでPhase 2設計だけに用いています。対応箇所・commit・license確認は [PAPER_MAP.md](PAPER_MAP.md)、相違は [PAPER_FIDELITY.md](PAPER_FIDELITY.md)。

今回の小規模比較はdevelopment評価です。memoryの有効性や未知taskへの汎化、完全な論文再現を示す実験ではありません。5つのOpenEvolve runは同じcircle-packing系taskで、独立な5課題ではありません。単純なtask名比較では全重複を検出できません。原文に未来の集計が紛れ込む問題はsource監査が必要です。

生成物はすべて `outputs/` に保存してGit除外します。新directory外の既存コード・DB・raw data、global環境、`~/.codex`、shell初期化ファイルは変更していません。

## 実行済みdemo結果（2026-09-17）

初回実装時の検証study: `outputs/autoresearch/demo-20260917T192545-27636`（今回の基盤変更後は継続不可。旧manifestは保持）。入力全612node（OpenEvolve 587＋SWE-smith 25）、各policyは1runにつき8要求。memoryはlocal 3 runから20 atoms＋4 skills。

| Policy | 平均rank attainment | 平均objective |
|---|---:|---:|
| random | 0.81444 | 0.80444 |
| dfs | 0.78587 | 0.77587 |
| bfs | 0.78587 | 0.77587 |
| current_best | 0.89059 | 0.88059 |
| meta_free | 0.80895 | 0.79895 |
| meta_memory | 0.80895 | 0.79895 |
| candidate v2（採用） | 0.89059 | 0.88059 |

平均はscoreのある5 runだけ。SWE-smith 1 runもreplayしたが数値平均から除外。すべてmean_rounds=8。

| Candidate | 仮説 | Objective | 採否 |
|---|---|---:|---|
| v1 | 観測rank＋depth加点 | 0.79895 | keep |
| v2_quality_first | depth加点を除き観測qualityを優先 | 0.88059 | keep |
| v3_memory_evidence | qualityにmemoryの証拠を加点 | 0.84773 | discard |

**現在のcandidateは採用済みv2へ戻してあります。** 固定baselineのmemoryあり/なしは同じ結果でした。v3のmemory加点はv2より悪化したため不採用です。memoryの有効性はこのdemoから確認できません。

これは同じdevelopment集合での比較です。候補を改善する動作を確認したもので、未知taskへの汎化や論文の性能再現を示す結果ではありません。小さい11nodeのrunは開始rootが最良で、rank=1だけでは探索改善の証拠になりません。

生成物: [demo log](outputs/demo.log)、[test log](outputs/tests.log)、[採否履歴](outputs/autoresearch/demo-20260917T192545-27636/results.jsonl)、[採用版の全結果](outputs/autoresearch/demo-20260917T192545-27636/trials/v2_quality_first/result.json)。
