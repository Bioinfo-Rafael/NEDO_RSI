# Evo-MCTS

Status: **partial**

## 取得内容

指定paper_data JSONL全38件、全5 production run logs、breakthrough JSON全6件。元物理dataset・モデル・評価器実行は取得/実行しない。

## Raw format / tree

paper JSONLはeval_times/depth/operator/thinking/reflection/code/fitness/algorithm。READMEはcomplete treeと呼ぶがparent/children/edgesは含まれない。logsの評価行とAction/Father Obj/Now Obj行を照合する。

## 共通schemaへのmapping

run=production log、node=eval_times、proposal=operator + 保存済みthinking/reflection、prompt=保存済みuser/system prompt（ある場合）、next state=保存済みcodeまたは評価結果。scoreはlogのobjectiveをそのまま保存（minimize）。paperの正のfitnessはmetrics_jsonに別保存し、符号反転しない。

## 実装根拠

repos/evo_mcts/source/evo_mcts.py: expand/run/serialize_mcts_tree; evolution_interface.py: get_algorithm/evolve_algorithmのnp.round(objective,5); mcts.py: MCTSNode.parent。

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 5 |
| Nodes | 2,101 |
| Edges | 673 |
| parent=NULL | 1,428 |
| Max stored depth | 9 |
| 2 children以上のnode | 85 |
| 最大children数 | 27 |
| Raw events | 2,818 |
| Parent coverage (edges/nodes) | 32.03% |
| Action coverage | 100.00% |
| Result coverage | 100.00% |
| Score coverage | 61.30% |
| Unknown parent relations | 1,381 |

## 復元不能項目・制約

Father Objを公式と同じ5桁丸めで照合し、先行する同run・depth-1の候補が一意の場合だけparentを復元。restore_mcts_treeの「最後の同depth node」heuristicは使用しない。infはREAL scoreに入れずNULL、元のinfはrawに保存。未公開rootや欠落codeを創作しない。depthは保持edgeから計算し、元depthはmetadata.original_depth。

## License

GPL-3.0（repository）

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `evomcts_audit.json`。

## Raw records found

2,101 records（単位: evaluation log record）。paper/breakthroughは既存evaluation nodeへの追加payload。

## Integrity / provenance

Raw provenance coverage: 100.00%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{"0": 2016, "12": 2, "13": 3, "15": 3, "16": 1, "18": 1, "2": 3, "20": 2, "27": 1, "3": 1, "4": 7, "5": 9, "6": 15, "7": 13, "8": 15, "9": 9}`。
