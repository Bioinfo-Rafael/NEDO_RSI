# Dream-RSI Experience Store

探索木を人が見る場合は`experience.xlsx`の`Experiences`シートを開いてください。
`run_id`でfilterし、`node_id`、`parent_id`、`depth`、`display_path`を読むと木を追えます。

ターミナルでtree表示する場合:

```bash
python3 dream_rsi_experience_store/scripts/print_trees.py
```

詳細な説明は`../dream_rsi_experience_store/README.md`にあります。

このdirectoryの正本は`experience.db`です。Dream-RSI replayで扱いやすいよう、DBを
`runs`、`experiences`、`raw_events`の3 tableに限定しています。

```text
Run
  ↓
Experience / DiscoveryNode
  ↓
Raw Event
```

## Data Model

### runs

1 row = 1 Discovery Treeです。`root_node_id`がtreeの開始nodeを示します。

### experiences

1 row = 1 DiscoveryNodeです。`node_id`と`parent_id`で木を表現します。proposal、score、
metrics、artifact、commit、trace範囲は同じrowにあります。

### raw_events

1 row = 1 raw Codex/autoresearch eventです。

```text
raw_events.node_id → experiences.node_id
```

割当できないeventは`node_id=NULL`です。Nodeを持たないraw sessionのeventも削除せず、
`run_id=NULL`で保持します。

## Replay query

```sql
SELECT * FROM runs;

SELECT * FROM experiences
WHERE run_id = ?
ORDER BY sequence_index;

SELECT * FROM experiences
WHERE parent_id = ?;

SELECT * FROM raw_events
WHERE node_id = ?
ORDER BY sequence_no, source_file, source_line;
```

## Python

```python
import json
import sqlite3

db = sqlite3.connect("experience_store/experience.db")
db.row_factory = sqlite3.Row

nodes = db.execute(
    "SELECT * FROM experiences WHERE run_id = ? ORDER BY sequence_index",
    ("run_660e45cf1cdc2d1538b777eb",),
).fetchall()

for node in nodes:
    metrics = json.loads(node["metrics_json"])
    print(node["node_id"], node["parent_id"], node["score"], metrics.get("objective"))

events = db.execute(
    """SELECT * FROM raw_events
       WHERE node_id = ?
       ORDER BY sequence_no, source_file, source_line""",
    (nodes[0]["node_id"],),
).fetchall()

db.close()
```

`experience.xlsx`は同じ内容を`Runs`、`Experiences`、`Raw_Events`の3 sheetで表示します。
設計と再生成方法の詳細は`../dream_rsi_experience_store/README.md`を参照してください。
