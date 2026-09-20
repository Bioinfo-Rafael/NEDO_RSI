# 再実行手順

作業rootはNEDO_RSI。Python 3.11以上、抽出・監査・DB buildには標準ライブラリだけを使う。既存DBは`mode=ro`と`query_only`で読む。bytecodeも既存directoryへ書かない。

## 元Experience DBの再構築

DATASET_BUILD.mdにsource別mappingと取得/変換commandを記載。固定取得manifestはsource_manifest.json。`rebuild_sources.py prepare` は新しい `outputs/rebuild` へ現在の変換コードをコピーし、hashを照合する。既存の同directoryがある場合は上書きせず停止する。

`download` は記録SHAに固定したGit checkout、固定HF revision、issue添付、Drive ZIPだけを新workspaceへ取得する。`prepare_raw.py`の後に`verify-raw`で全rawのSHA-256を確認する。旧rawと同一のfileが取得できないときは再現成功としない。このタスクでは再ダウンロードや45GB DBの再生成は実行していない。

localの旧ingestと現DDLの不整合はDATASET_BUILD.mdに記載。手元のlocal DBを新workspaceへ読み取り専用backupするには:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 branch_memory_store/rebuild_sources.py prepare
PYTHONDONTWRITEBYTECODE=1 python3 branch_memory_store/rebuild_sources.py local-snapshot
```

この2commandは今回、新directory内で実行確認済み。既存local DBは不変。snapshotはrawからの再計算とは区別する。第三者はこのlocal DB入力を別途入手するか、旧DDLと全rawを回収する必要がある。

## 既存DBからbranch memoryへ

```bash
cd /Users/cls-lab/Git/NEDO_RSI
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/branch_memory_store/src"

# DB行数、分岐、column/table容量、source hashを読み取り専用で計測
python3 -m branch_memory.cli audit

# 分岐のあるrunを全件圧縮。既存units.jsonlがあれば上書きせず停止。
python3 -m branch_memory.cli extract

# 固定seedで100件選ぶ
python3 -m branch_memory.cli select --limit 100 --seed 17 --max-chars 100000
```

exportはこのshell process内だけの設定。global shell設定やlaunchctlは変更しない。既存生成物がある場合、auditは同じsourceの保存済み測定を再利用し、extractは省略する。新source snapshot用に監査結果を上書きして旧結果と混ぜない。

## Codex認証とpilot

```bash
mkdir -p branch_memory_store/outputs/codex-home
env CODEX_HOME="$PWD/branch_memory_store/outputs/codex-home" \
  codex -c 'cli_auth_credentials_store="file"' login

python3 -m branch_memory.cli delex --limit 10 --model gpt-5.6-sol
```

今回は既存NEDO専用認証の読み取り用コピーを専用homeへ置き、認証元を変更せず利用した。秘密はコード・manifest・Git管理fileへ書かない。上のloginは別環境で認証を用意する手順。出力cacheはprompt/model hashごとに保持するため、今回の既存completedをそのまま再処理しない。

pilot結果を見てprompt/選定を確定後:

```bash
# 未完了unitを最大100件。各callは1unit、最大4並列。
python3 -m branch_memory.cli delex --limit 100 --model gpt-5.6-sol --workers 4
python3 -m branch_memory.cli build --model gpt-5.6-sol
python3 -m branch_memory.cli verify --full-hash
```

5GBはdecimal `5_000_000_000` bytes。SQLite max_page_countとexport直前のfilesizeで二重に上限検査する。新DBはruns/branch_pointsの2table。build時に使用するmodelを変えると、そのmodelで完了したcacheだけを反映する。未完了を捏造せずpendingとして残す。

## Excel

```bash
python3 -m branch_memory.cli excel-data
# Codex同梱runtimeの例。別環境では同等のartifact-tool環境を用意する。
ln -s "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules" \
  branch_memory_store/node_modules
"$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node" \
  branch_memory_store/export_excel.mjs
```

node_modulesリンクがすでにあれば作成commandを省略する。依存directoryの内容は変更しない。ExcelはBranch_Points / Runs / Statsの3sheet、全selected rowを収録。長文セルはExcelの制限を超える場合のみ全文DB参照＋previewに置き換え、そのcell一覧はoutputs/excel_long_cells.json。raw全体はDBで保持する。表示のrow高さに収まらない長文はExcelの数式バーで読む。

## 検証・coverage・report

```bash
python3 -m unittest discover -s branch_memory_store/tests -v
python3 -m branch_memory.cli coverage \
  --source-db experience_store/experience.db \
  --node-id node_50730af5db97b538085969fe
python3 -c 'from branch_memory.reports import measured_reports; measured_reports()'
```

coverageはselected owner / discarded_sampling / excluded_linear_run / unresolvedを元ID単位で照会する。referenceにnormalized DBを記録したpublic nodeはそのsource DBを指定する。public/combinedを二重入力にしない。

## 再現性の範囲

同じ元DBからの抽出、割当、選定、source hashは決定論的。モデル出力は同じmodel名・promptでもbit単位の再現を保証しない。取得済みcacheを保持すれば同一delex JSONを使ってDBを再生成できる。SQLiteファイルやExcelのZIP metadataのbyte一致より、ID・raw・delex・hashの論理一致を検査する。
