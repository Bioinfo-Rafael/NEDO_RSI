# Search Agents

Status: **partial**

## 取得内容

READMEの公開Google Drive gpt4o_search_trajectories.zip（2,189,522,114 B）を認証なしで取得。全910 HTMLとresults.txt。sitesはshopping/reddit/classifieds（VisualWebArena）。別WebArena archiveへの公開リンクは確認できなかった。

## Raw format / tree

RenderHelperのstate_obv/predict_action/additional_textをparse。候補にはa_idx,curr_a_idx,depth,scoreがある。選択済みbest_actionsは複数pageに同じcandidate一覧が繰り返されるため同じbatch内でdedup。

## 共通schemaへのmapping

prompt=実行pageの観測、proposal=raw_prediction、result_summary=選択action後の次page観測、score=sourceのvalue。初期観測は実観測nodeとして保持。task PASS/FAILはrun metadataとnode_id=NULLのraw_eventsへ保存し、候補rewardにしない。

## 実装根拠

repos/search_agents/run.py: take_action_and_score、action_queue、all_candidates、best_actions; browser_env/helper_functions.py: RenderHelper.render。

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 892 |
| Nodes | 18,099 |
| Edges | 11,878 |
| parent=NULL | 6,221 |
| Max stored depth | 6 |
| 2 children以上のnode | 2,387 |
| 最大children数 | 10 |
| Raw events | 21,800 |
| Parent coverage (edges/nodes) | 65.63% |
| Action coverage | 93.49% |
| Result coverage | 17.69% |
| Score coverage | 92.92% |
| Unknown parent relations | 5,329 |

## 復元不能項目・制約

depth0の親はsearch開始時の実行node。深い候補は同a_idx・直前depthの親候補が一意のときのみ接続。curr_a_idxはlocal sibling indexでglobal parent IDではない。曖昧ならparent=NULL/parent_relation=unknown。未選択branchの観測は保存されていない。空HTMLはrawに残すがnodeを作らない。

## License

MIT（repository）。Drive archive内の独立したlicenseは未確認

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `searchagents_audit.json`。

## Raw records found

910 records（単位: render HTML）。page内のcandidateも展開するためnode数とは一致しない。

## Integrity / provenance

Raw provenance coverage: 100.00%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{"0": 13814, "1": 1898, "10": 113, "2": 576, "3": 378, "4": 314, "5": 884, "6": 39, "7": 29, "8": 35, "9": 19}`。
