# ReST-MCTS

Status: **partial**

## 取得内容

README Data & Modelの全20 Hugging Face datasets、全25 JSON files。PRM-0th/PRM-1st、3model×2round×3methodのpolicy/CoT/DPOを取得。モデルweightは取得しない。

## Raw format / tree

JSON配列、拡張子.jsonのJSONL、DPO column arrays。instruction/output、content/summary、prompt_answer/label、chosen/rejected、response_chosen/response_rejectedを扱う。

## 共通schemaへのmapping

1公開path=1 run。DPOはchosen/rejectedを別runとしchoiceをmetadataに保持。明示的Step n:の境界だけでnodeへ分割。prompt=question+prefix、proposal=step、result_summary=prefix+step。PRM labelは最終prefix nodeだけに保存し、中間stepへ配らない。無記録scoreはNULL。

## 実装根拠

repos/rest_mcts/MCTS/base.py: treeNode.append_children/update_y_from_parent/get_full_value_samples*; MCTS/task.py: policy_samples/value_samples; utils/format_dpo.py; PRM/train_VM_chatglm.py。

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 1,934,674 |
| Nodes | 5,015,746 |
| Edges | 3,081,072 |
| parent=NULL | 1,934,674 |
| Max stored depth | 36 |
| 2 children以上のnode | 0 |
| 最大children数 | 1 |
| Raw events | 5,015,746 |
| Parent coverage (edges/nodes) | 61.43% |
| Action coverage | 99.64% |
| Result coverage | 99.64% |
| Score coverage | 22.87% |
| Unknown parent relations | 0 |

## 復元不能項目・制約

公開training pathsから元MCTSの完全treeや共通ancestor IDは復元できない。同じquestion/prefixでも異なる探索runを勝手にmergeしない。PRMはsourceが付けたprocess labelであり、今回生成したrewardではない。CoT/DPO baselineはlinear_reasoningとしてMCTSと区別。空solutionにもlabelがある場合、action/resultはNULL。重複する公開dataset/fileは別出典として保持しており、独立な実験数ではない。

## License

PRM-0th: Apache-2.0。他19 datasets: CC-BY-4.0（各HF cardData）。code repositoryのlicenseは未確認

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `restmcts_audit.json`。

## Raw records found

1,792,677 records（単位: published JSON/JSONL row）。DPO rowからchosen/rejected各1 run。各pathを明示stepに分割。dataset間重複を含む。

## Integrity / provenance

Raw provenance coverage: 100.00%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{"0": 1934674, "1": 3081072}`。
