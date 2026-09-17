# SWE-agent / SWE-smith

Status: **obtained**

## 取得内容

SWE-agent v1.1.0 release→SWE-smith→公式HF trajectoriesを追跡。tool/xml/ticks/trainの全32 parquet、repoの18 demonstration .trajを取得。repository snapshot/Docker imageは取得しない。

## Raw format / tree

.trajのtrajectory[thought,action,observation]、HF messages/instance_id/resolved/model/traj_id/patch。messagesはtool/xml/ticksでJSON文字列、trainで配列。traj_idが同じencoding違いはtool→xml→ticks→trainの順で1つ採用し、すべての元encodingはrawに残す。

## 共通schemaへのmapping

1 trajectory=1 run、1 assistant action turn=1 node。parentは直前turn。prompt=直前observation（初回はissue/system context）、proposal=tool_callsまたはassistant content、result_summary=次のassistantまでのobservations。prefix全体はmetadata内のfile,row,message indexで参照する。

## 実装根拠

repos/swe_agent/docs/usage/trajectories.md、sweagent/agent/agents.pyのtrajectory記録。HFのpinned README/featuresと全parquet schemaを確認。

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 49,915 |
| Nodes | 1,508,707 |
| Edges | 1,458,792 |
| parent=NULL | 49,915 |
| Max stored depth | 150 |
| 2 children以上のnode | 0 |
| 最大children数 | 1 |
| Raw events | 1,508,707 |
| Parent coverage (edges/nodes) | 96.69% |
| Action coverage | 100.00% |
| Result coverage | 96.69% |
| Score coverage | 0.00% |
| Unknown parent relations | 0 |

## 復元不能項目・制約

resolvedはbooleanのまま最終node.metrics_json/run metadataに保存し、新しい0/1 rewardにしない。公開sourceに数値scoreがないためscore=NULL。最後のsubmit後のobservationが未保存ならNULL。返されたsource patchを検証・修正・実行していない。

## License

MIT（repository/Hugging Face dataset）

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `sweagent_audit.json`。

## Raw records found

102,078 records（単位: parquet trajectory row）。traj_idでencoding重複を除外。demonstrationは別run。

## Integrity / provenance

Raw provenance coverage: 100.00%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{"0": 49915, "1": 1458792}`。

## 元の位置と正規化ID

SWE-smithの元messagesにnode/parent IDはありません。`node_id`末尾と`metadata.original_parent_id`はassistant turnの0始まり番号、`metadata.message_index` / `original_node_id`は元messages配列の位置です。親の元messages位置は、`parent_id`で親nodeを読み、その`message_index`を参照します。結合には共通列の`node_id` / `parent_id`を使用してください。
