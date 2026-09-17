# 公開探索履歴 Experience Store

公開された探索履歴を、既存Experience Storeと同じ3テーブルへ変換したものです。モデル学習やLLM API呼び出しは行いません。

```text
Public search histories → source adapters → Runs / Experiences / Raw_Events → public_experience.db
```

- `Run` = 1本の探索履歴 / trajectory
- `Experience` = 1 node、`parent_id` = 復元できた親へのedge
- `prompt` = state/context、`proposal` = action、`result_summary` = next state/outcome
- `score` = sourceに記録された数値のみ。欠落はNULL

## 読むファイル

- `public_experience.db`: 公開データ全件
- `combined_experience.db`: 元の自前データを全columnそのままcopy + 公開データ
- `public_experience.xlsx` / `combined_experience.xlsx`: `Runs / Experiences / Raw_Events`の3シート。run単位サンプル。全件・収録・省略件数は[Excel範囲](reports/excel_sample.md)
- [木のサンプル](reports/tree_samples.txt) / [最終監査](reports/final_audit.md) / [取得元・license](reports/source_inventory.md)

## 変換結果

| Source | Status | Runs | Nodes |
|---|---|---:|---:|
| [OpenEvolve](reports/openevolve.md) | obtained | 5 | 587 |
| [Evo-MCTS](reports/evo_mcts.md) | partial | 5 | 2,101 |
| [ReST-MCTS](reports/rest_mcts.md) | partial | 1,934,674 | 5,015,746 |
| [SWE-agent / SWE-smith](reports/swe_agent.md) | obtained | 49,915 | 1,508,707 |
| [Search Agents](reports/search_agents.md) | partial | 892 | 18,099 |
| [Dream-RSI](reports/dream_rsi.md) | unavailable | 0 | 0 |
| [Tree of Thoughts（追加）](reports/treeofthoughts.md) | obtained | 240 | 13,268 |

取得済みの履歴にも欠落があります。`parent_relation=unknown`は親不明であり真のrootとは限りません。`depth`は保存したedgeからの深さ、元depthはmetadataに保持します。複数rootのrunでは`root_node_id=NULL`、root一覧はrun metadataです。ReSTはpath、SWEはlinearで、完全な分岐探索木と同一視しないでください。

## データモデル：3テーブルと全カラム

`public_experience.db` と `combined_experience.db` は同じ3テーブル構成です。定義は [src/schema.sql](src/schema.sql) にあります。以下は実DBの `sqlite_master`、`PRAGMA table_info`、`PRAGMA foreign_key_list` と照合した仕様です。

| テーブル | 1行が表すもの | 主キー | カラム数 |
|---|---|---|---:|
| `runs` | 1回の探索・実行履歴 | `run_id` | 14 |
| `experiences` | 探索木の1ノード、または実行の1ステップ | `node_id` | 32 |
| `raw_events` | メッセージ・ツール実行などの元イベント | `event_id` | 19 |

### runs：実行全体の情報

| カラム | 内容 |
|---|---|
| `run_id` | 実行を識別するID。主キー |
| `project_name`, `task_name` | プロジェクト名・タスク名 |
| `root_node_id` | 起点となるノードID。複数起点の場合はNULL |
| `status` | 実行の状態 |
| `start_time`, `end_time` | 開始・終了時刻 |
| `replay_eligible` | replay対象かを示す0/1のフラグ |
| `source_type`, `source_reference` | データの種類と出典 |
| `git_branch` | 関連するGitブランチ |
| `codex_thread_ids_json` | 関連するCodex thread IDの一覧 |
| `original_session_ids_json` | 元データのsession IDの一覧 |
| `metadata_json` | 構造の種類・起点一覧などの追加情報 |

### experiences：ノードの情報

| カラム | 内容 |
|---|---|
| `run_id` | 所属する `runs.run_id` |
| `node_id` | ノードID。主キー |
| `parent_id` | 親の `experiences.node_id` |
| `branch_id` | ブランチの識別情報 |
| `attempt_index` | 試行番号 |
| `sequence_index` | 表示・整列用の順番。親子関係を表すものではない |
| `depth` | 保存した親子関係をたどった深さ。起点は0 |
| `display_path` | 親子関係に沿ったノードのパス |
| `node_type` | ノードの種類 |
| `proposal` | 提案・行動の内容 |
| `prompt` | 入力・状況・文脈 |
| `result_summary` | 結果・観測の要約 |
| `status` | ノードの状態 |
| `score` | 元データに存在する数値スコア。欠落はNULL |
| `score_name`, `score_direction` | スコア名と方向。方向は `maximize` / `minimize` / `unknown` |
| `metrics_json` | その他の評価指標 |
| `artifacts_json` | 成果物の情報 |
| `commit_hash`, `commit_hashes_json` | 関連コミットとコミット一覧 |
| `commit_message` | コミットメッセージ |
| `git_commit_before`, `git_commit_after` | 変更前後のコミット |
| `event_count` | 紐づくイベント数 |
| `trace_start_time`, `trace_end_time` | ログの開始・終了時刻 |
| `first_event_sequence`, `last_event_sequence` | 対応するイベント範囲 |
| `source_type`, `source_reference` | データの種類と出典 |
| `source_files_json` | 元ファイルの一覧 |
| `metadata_json` | 親子関係の根拠などの追加情報 |

### raw_events：元ログの情報

| カラム | 内容 |
|---|---|
| `event_id` | イベントID。主キー |
| `run_id` | 関連する実行ID |
| `node_id` | 紐づくノードID |
| `sequence_no` | イベントの順番 |
| `timestamp` | 時刻 |
| `event_type` | イベントの種類 |
| `role` | 発言・実行主体の役割 |
| `tool_name` | ツール名 |
| `command`, `cwd` | 実行コマンドと作業ディレクトリ |
| `stdout`, `stderr`, `exit_code` | 標準出力・標準エラー・終了コード |
| `message` | メッセージ本文 |
| `source_file`, `source_line` | 元ファイルと行・レコード位置 |
| `binding_method`, `binding_confidence` | ノードへの紐づけ方法と確信度 |
| `raw_json` | 元イベントのJSON |

`*_json` と `raw_json` はJSON文字列を格納するTEXT型です。Pythonでは `json.loads()`、SQLiteでは `json_extract()` で読み取れます。数値列はスキーマでINTEGERまたはREAL、その他の列はTEXTです。元データにない情報はNULLや空のJSON配列・オブジェクトになります。ノードに紐づけられない元イベントでは `node_id` がNULLになる場合があります。

### テーブル間の参照関係

| 参照元 | 参照先 | 種類 | 関係・意味 |
|---|---|---|---|
| `experiences.run_id` | `runs.run_id` | Physical FK | N:1。ノードが所属する実行 |
| `experiences.parent_id` | `experiences.node_id` | Physical FK・自己参照 | N:0..1。各ノードの親。親から見れば子は複数可 |
| `raw_events.node_id` | `experiences.node_id` | Physical FK | N:0..1。イベントが紐づくノード |
| `runs.root_node_id` | `experiences.node_id` | Logical Reference | 各runから0または1件の起点を参照 |
| `raw_events.run_id` | `runs.run_id` | Logical Reference | N:0..1。イベントに関連する実行 |

Physical FKはSQLiteスキーマに宣言された外部キー、Logical ReferenceはFK制約がなくコード上で対応づける参照です。SQLiteで外部キー制約を実際に強制するには、接続ごとに `PRAGMA foreign_keys=ON` が必要です。

```text
runs：1回の探索・実行
 └─ experiences：その実行に属するノード群
      ├─ parent_id：同じテーブル内の親ノードへの参照
      └─ raw_events：そのノードに紐づく元ログ
```

### linearとtreeの保存方法

**linear用・tree用の別テーブルやedgeテーブルはありません。`experiences.parent_id` が辺を表します。** 以下のIDは説明用に短縮した例です。

#### linear：一本道

```text
N1 → N2 → N3
```

| node_id | parent_id | sequence_index | depth |
|---|---|---:|---:|
| N1 | NULL | 0 | 0 |
| N2 | N1 | 1 | 1 |
| N3 | N2 | 2 | 2 |

確認したSWE-smithの実行履歴は、このように前のステップを親にする構造です。

#### tree：分岐がある

```text
N1
├─ N2
│  └─ N4
└─ N3
```

| node_id | parent_id | sequence_index | depth |
|---|---|---:|---:|
| N1 | NULL | 0 | 0 |
| N2 | N1 | 1 | 1 |
| N3 | N1 | 2 | 1 |
| N4 | N2 | 3 | 2 |

**同じ `parent_id` を持つノードが兄弟です。** 確認したOpenEvolveの履歴にも同じ起点から複数の子が分岐するデータがあります。`sequence_index` は表示順で、親子関係は `parent_id` から読み取ります。`branch_id` が空でも分岐は表現できます。

公開runの分類は `runs.metadata_json` 内の `source_structure` に保存します。独立したカラムではありません。

```json
{"source_structure": "tree"}
```

`linear` / `tree` のほか、元データに応じて `linearized_mcts`、`linear_reasoning`、`partial_tree`、`partial_search_tree`、`beam_search_forest`、`dfs_search_forest` などがあります。分類だけから完全な探索木が復元できたと判断せず、保存された辺と取得元の制約を確認してください。

- 親は1つなので、複数の親が合流するDAGを単一ノードで直接表現できません。
- `parent_id=NULL` は真の起点だけでなく、元データから親を復元できない場合も含みます。
- 親子関係の根拠は `experiences.metadata_json.parent_relation` に記録します。`explicit`、`reconstructed_from_official_code`、`linear_sequence`、`unknown` を区別します。
- 複数起点の場合は森になります。`runs.root_node_id` はNULL、起点一覧は `runs.metadata_json.root_node_ids` に保存します。
- `source_dataset` もJSON内の情報です。Excelで表示用カラムとして展開されていても、DBの独立カラムではありません。

## 探索木を見る

NEDO_RSI直下で実行します。まずrun IDを選びます。

```bash
sqlite3 public_experience/public_experience.db 'SELECT run_id,source_type FROM runs LIMIT 10;'
python3 dream_rsi_experience_store/scripts/print_trees.py \
  --db public_experience/public_experience.db \
  --run-id openevolve::MyOpenEvolve::21d972d0-0b7d-49dd-8f88-f98a26dbc6ee
```

全runの無指定表示は出力が巨大になるため、`--run-id`を付けてください。GoT Viewer用got.jsonへの変換は今回行っていません。

## Pythonから読む

```python
import json, sqlite3
db = sqlite3.connect("file:public_experience/public_experience.db?mode=ro", uri=True)
db.row_factory = sqlite3.Row
run_id = db.execute("SELECT run_id FROM runs WHERE run_id GLOB ? ORDER BY run_id LIMIT 1", ("openevolve::*",)).fetchone()[0]
for node in db.execute("SELECT * FROM experiences WHERE run_id=? ORDER BY sequence_index", (run_id,)):
    meta = json.loads(node["metadata_json"])
    print(node["node_id"], node["parent_id"], node["score"], meta["parent_relation"])
db.close()
```

SQLiteはPython標準sqlite3だけで読めます。公開行のsource_file/source_files_jsonの相対pathはpublic_experience/基準です。combinedにコピーした既存local行のpathは元仕様のまま保持しています。元データへの場所はsource_reference/source_files_json、より細かいJSON/parquet row/HTML page位置はmetadata_jsonにあります。raw_events.source_lineはJSONLでは行、JSON配列/parquet/HTMLでは1始まりのrecord/page番号です。

## 再現とparser改善

```bash
cd public_experience
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python src/autoresearch.py --label next-candidate --source openevolve
.venv/bin/python src/build_db.py --source openevolve
```

取得計画は[download_plan.md](reports/download_plan.md)。inventory.py / external_inventory.pyは取得前metadata、download.py / download_search_agents.py / prepare_raw.pyはraw取得。全sourceのDB生成後は `src/build_db.py --combine` → `src/prepare_excel.py` → `src/export_excel.mjs` → `src/validate_excel.py` → `src/reports.py` の順です。Python scriptsは`.venv/bin/python`で実行します。Excel出力はCodex runtimeのartifact-toolを使い、`node src/export_excel.mjs public_experience` と `node src/export_excel.mjs combined_experience` を実行します。モデルやsource内の実験コードは実行しません。

parserの変更許可範囲・評価方法は[program.md](program.md)、試行記録は[autoresearch_report.md](reports/autoresearch_report.md)。

既存 `experience_store/experience.db` / `.xlsx` はread-onlyで、SHA-256が一致しています。今回のraw/repos/DBはlocal Git exclude対象で、この作業ではpushしていません。
