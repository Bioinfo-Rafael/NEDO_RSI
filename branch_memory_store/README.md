# Branch Memory Store

**既存の探索履歴から分岐の意思決定点だけを残し、その前後の実験・行動をCodexでtask-independentな表現へ変換したmemory datasetです。** 1 rowは1 branching pointです。原文と抽象化結果を並べて保持し、探索policyが「似た分岐状況で、どのような探索方向と結果があったか」を検索するために使います。

```text
既存Experience DB（読み取り専用）
  → known children >= 2 のnode
  → incoming / terminal / next branchへ圧縮
  → 固定seedによる選定
  → Codexのsemantic abstraction
  → branch_memory.db / branch_memory.xlsx
```

既存6,558,531 nodeのうち、復元された分岐は5,326点（946 run）。そのうち100点を選定しました。linear-only runは新memoryに含めません。元DB・raw・コードを変更せず、新しい生成物だけをこのdirectoryに保存します。

## 成果物と読む順序

- `branch_memory.db`: 正本。SQLiteのruns / branch_pointsの2table。rawとdelexを保持。
- `branch_memory.xlsx`: Branch_Points / Runs / Statsの3sheet。選定した全rowを読むための表示用export。
- [DATASET_BUILD.md](DATASET_BUILD.md): 旧DBの取得先・固定revision・raw→canonicalの全source mapping・再構築手順。local旧ingestと現DDLの不整合も明記。
- [STORAGE_AUDIT.md](STORAGE_AUDIT.md): 実DBのtable/index/column別容量、source別件数。
- [DELEXICALIZATION.md](DELEXICALIZATION.md): B/F例、圧縮規則、prompt、出力schema、retry/cache。
- [SELECTION_REPORT.md](SELECTION_REPORT.md): source別の採用・除外件数と理由。
- [REPRODUCE.md](REPRODUCE.md): 実際の再実行command。
- [PILOT_REPORT.md](PILOT_REPORT.md): pilotの時間・token量・品質確認と、最終100件の実測結果。

## 今回の生成結果（2026-09-20）

| 項目 | 結果 |
|---|---|
| branch_memory.db | 3,690,496 bytes（約3.69 MB）、上限5,000,000,000 bytes |
| Table rows | runs: 57、branch_points: 100 |
| Semantic処理 | 全100件completed、failed 0 |
| Model / prompt | gpt-5.6-sol / branch-semantic-v2 |
| 再開確認 | 100件cache hit、追加model call 0 |
| branch_memory.xlsx | 240,986 bytes、3sheet。2,297入力cellの保存値一致を確認 |
| 構造・安全性 | 17 tests passed、圧縮tree cycle 0、重複node割当0 |

元のcombined DBは45,615,620,096 bytes。新DBは全履歴の可逆圧縮ではなく、5,326分岐点から固定seedで選定した100点のmemoryである。選外は5,226点（input_size_limit: 182、stratified_sampling: 5,044）。元DB10個の全ファイルSHA-256が一致し、original DB modified countは**0**。保護対象12,747ファイルにも変更なし。結果はoutputs/verification.json。

**旧DBをrawから完全再構築する際の制約:** localの旧ingestは現在の3table DDLに対応していない。public側の固定取得・変換手順を文書化し、combined生成には既存local DBのsnapshotを入力する手順を用意した。旧DDLや非公開rawがない第三者について、local部分までゼロから再現可能とはしていない。

## DBの使い方

```python
import json
import sqlite3
from pathlib import Path

path = Path('branch_memory_store/branch_memory.db').resolve()
db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
db.row_factory = sqlite3.Row
rows = db.execute('''
    SELECT branch_point_id, source_type, raw_context_json, delexicalized_json
    FROM branch_points
    WHERE delex_status='completed'
      AND json_extract(delexicalized_json, '$.search_pattern') LIKE ?
    LIMIT 5
''', ('%refinement%',))
for row in rows:
    abstract = json.loads(row['delexicalized_json'])
    raw = json.loads(row['raw_context_json'])
    print(row['branch_point_id'], abstract['search_pattern'])
    print('原文:', raw['branch_point']['proposal'])
db.close()
```

上は単語検索例で、embedding/vector indexは追加していません。検索後はincoming/terminalのrawとscoreの出典を確認してください。search_patternはこの履歴での観測であり、一般則・因果・次の実験の成功保証として扱いません。

## 2tableのデータモデル

| Table / primary key | 単位 | 役割 |
|---|---|---|
| runs / run_id | selected branchを含む元run | source DB/task/source reference、共有root anchor、元run metadata |
| branch_points / branch_point_id | 1分岐点 | 元node ID・圧縮関係・前後trajectory・rawとsemantic abstraction・処理provenance |

| branch_pointsのcolumn群 | 意味 |
|---|---|
| run_id | **Physical FK** → runs.run_id、N:1 |
| branch_point_id / original_node_id | 元canonical experiences.node_idを維持。元source DBへのlogical reference |
| previous_branch_point_id / next_branch_point_ids_json | **Logical self reference**。元圧縮treeの直前/直後の分岐。samplingで相手が未収録のことがある |
| incoming_node_ids_json / incoming_raw_json | 直前分岐/rootと現在分岐の間の一本道。両端を除く |
| terminal_branch_node_ids_json / terminal_branches_raw_json | 次分岐まで進まずleafで終わる各経路 |
| branching_factor / depth | 実際の子数 / 復元できたcomponent rootからのedge数 |
| source_type / source_reference | source分類 / 元DB・run・node・raw locator |
| raw_context_json | 分岐点自身のrawとroot anchor ID、不完全性 |
| delexicalized_json / delex_status | 構造化された意味抽象化 / pending・completed・failedの処理状態 |
| delex_model / delex_prompt_version / prompt_sha256 | 使用modelと実promptの版・hash |
| source_hash / input_sha256 / output_sha256 | 圧縮raw unit / モデル入力 / 抽象化出力のSHA-256 |
| codex_version / timestamp / cache_key | CLI版、UTC生成日時、再利用cacheのキー |

schemaは `src/branch_memory/build.py:SCHEMA`。`runs`にはrun_id、source_type、task_name、source_db、source_reference、root_anchors_json、metadata_jsonの7列。`branch_points`は上の25列。previousがNULLなら、そのcomponentで先行分岐がないという意味。元探索の真のrootが公開されているとは限りません。

## 探索木を追う

```sql
SELECT branch_point_id, previous_branch_point_id,
       next_branch_point_ids_json, depth, branching_factor
FROM branch_points WHERE run_id = ? ORDER BY depth, branch_point_id;
```

next/previousのIDが新DBに存在しなければ選外の分岐です。`source_reference`で元DBへ戻れます。選外を飛ばして勝手に別の親へ付け替えません。全分岐の元圧縮関係はoutputs/units.jsonlに保存しています。

## 利用範囲

最終DBはdecimal 5GB以内で検査します。容量を満たすためのpaddingや巨大raw event複製はしません。抽出・採番・選定は決定論的ですが、semantic JSONはモデルの解釈を含みます。全件の完全な意味正確性は保証しません。取得時の公開範囲、partial tree、推定parent、sample選定の偏りを保ったデータです。

本番autoresearchや研究コードの実験は開始していません。このdatasetを既存policyのretrievalへ接続する処理も独立の次工程です。

## 実データのrawとdelexの比較（3例）

以下の日本語は保存された英語JSONの要約。原文・ID・実数値はDBに保持している。

### 1. NEDO_RSI：同じ変更を別々に評価した分岐

ID: `node_50730af5db97b538085969fe`

- **Raw incoming:** 空。分岐点自身は`Initial synchronous linear baseline`、最小化scoreは1.0。
- **Raw branches:** `Smoke: pipeline transport only`を別々に評価した2経路。scoreは0.9278946103と0.9953474535。
- **Delex state:** 予測誤差と制御コストを合わせた目的で評価する、同期的な線形予測器のbaseline。
- **Delex strategy / outcome:** 同じpipeline transport変更の独立した評価。両経路の集約目的は改善したが、予測と制御のtradeoffは異なる。親関係は元DBの時系列推定であり、変更の意味や因果効果を確定していない。

### 2. OpenEvolve：timeout後の2種類の実装変更

ID: `openevolve::MyOpenEvolve::abf1c182-d6c4-4873-b57f-c79d968ce587::c0430a9d-e071-4991-a0a9-65d819708190`

- **Raw incoming:** 空。分岐点の評価はtimeout。
- **Raw branches:** 円配置programを、同心配置・短いpairwise force処理へ書き直す経路と、千鳥配置・vectorization・局所探索へ書き直す経路。terminalのscoreは0と0.0432356274。
- **Delex state:** 構造的初期化と反復的制約調整を使う配置最適化器が実行時間制約に達した状態。
- **Delex strategy / outcome:** 全体処理を単純化する方策と、計算をvectorizeして一部を局所改善する方策。両方に実行可能な評価結果がある。ただし親と同じ指標で比較できないため、baseline比のoutcome_typeは両方`unknown`。

### 3. Tree of Thoughts：一本道の履歴から次の3分岐へ

ID: `treeofthoughts::2aeea08ba9e8152a1889::6::32`

- **Raw incoming:** node末尾`::30`の`v5. eater`、`::31`の`v4. sieve`。分岐点自身は`v1. crush`。
- **Raw branches:** terminalなし。次の分岐点`::33`、`::65`、`::93`へのlink。
- **Delex state:** 部分的な構造解へ候補を順次追加した深さ優先探索の状態。
- **Delex strategy / outcome:** 次の3意思決定点へ探索を継続する構造。これらの先のactionや最終結果はこのunitに含まれないため、成功・失敗を補完しない。
