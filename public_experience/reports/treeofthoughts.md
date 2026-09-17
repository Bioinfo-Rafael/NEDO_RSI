# Tree of Thoughts（追加）

Status: **obtained**

## 取得内容

追加探索2候補中、princeton-nlp/tree-of-thought-llmを採用。全12 log/cache JSONを取得。変換対象は2 beam logsと2 DFS logsの4 files（240 search runs）。他8 filesはnaive baseline/cacheとしてrawにのみ保持。

## Raw format / tree

beam: steps[].ys/new_ys/values/select_new_ys。DFS: actions[],env_step,total_step,info.r_word/r_letter。

## 共通schemaへのmapping

beamはofficial生成処理がparent prefix+new textを返すことを確認し、直前levelの選択prefixが一意のときだけ親にする。DFSはactions[:-1]と一致するstackが一意なら親にする。score=valueまたはr_wordをそのまま保存、maximize。

## 実装根拠

repos/treeofthoughts/src/tot/methods/bfs.py: get_samples/get_proposals/solve; scripts/crosswords/search_crosswords-dfs.ipynb: dfsのactions.append/popとinfo保存。

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 240 |
| Nodes | 13,268 |
| Edges | 11,750 |
| parent=NULL | 1,518 |
| Max stored depth | 8 |
| 2 children以上のnode | 2,709 |
| 最大children数 | 46 |
| Raw events | 13,268 |
| Parent coverage (edges/nodes) | 88.56% |
| Action coverage | 100.00% |
| Result coverage | 100.00% |
| Score coverage | 100.00% |
| Unknown parent relations | 14 |

## 復元不能項目・制約

prompt-only rootはlogにないため人工nodeを追加しない。rootの子群はforestとして保持。重複prefixで親が一意に決まらない場合はNULL。LATSも調査したが、saved implementations/feedbackは全parentを保存しないうえaccは累積task精度なのでnode scoreとして採用しなかった。

## License

MIT（repository）

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `treeofthoughts_audit.json`。

## Raw records found

240 records（単位: search log top-level run）。4 search logs対象。別の8 naive/cache JSONはraw保管のみ。

## Integrity / provenance

Raw provenance coverage: 100.00%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{"0": 10098, "1": 461, "10": 28, "11": 10, "12": 27, "14": 4, "15": 1, "16": 1, "2": 344, "3": 1020, "38": 1, "4": 624, "46": 1, "5": 183, "6": 172, "7": 172, "8": 80, "9": 41}`。
