# OpenEvolve

Status: **obtained**

## 取得内容

issue #156の本文・commentsから解決したypwang61/MyOpenEvolveをSHA固定clone。全program checkpointを取得。公式repoのreleasesには履歴assetなし。checkpoint関連issue 23件も調査し、#88の単一programと#141のlogを追加取得。

## Raw format / tree

Program JSONのid/parent_id/generation/iteration_found/code/metrics/prompts。異なるcheckpointで同一UUIDは同一nodeにまとめ、promptsを最も多く保持するsnapshotを採用し、全snapshotをraw_eventsに残す。親リンクの連結成分を1 runとする。

## 共通schemaへのmapping

prompt=parent code/metrics、proposal=changes_descriptionまたは保存済みLLM responses、result_summary=child code、score=metrics.combined_score。元ProgramDatabase.get_best_programのdescending順によりmaximize。

## 実装根拠

repos/myopenevolve/openevolve/database.py: Program / _save_program / get_best_program; controller.py: child Program生成・log_prompts。

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 5 |
| Nodes | 587 |
| Edges | 582 |
| parent=NULL | 5 |
| Max stored depth | 16 |
| 2 children以上のnode | 143 |
| 最大children数 | 10 |
| Raw events | 3,933 |
| Parent coverage (edges/nodes) | 99.15% |
| Action coverage | 97.10% |
| Result coverage | 100.00% |
| Score coverage | 86.03% |
| Unknown parent relations | 0 |

## 復元不能項目・制約

5 root UUIDは別tree。snapshotを別nodeとして水増ししない。initial programsにはactionなし。combined_scoreのないnodeはNULL。#88/#141はrun/parentを確定できないためunbound raw_eventsとして保持。

## License

Apache-2.0（repository。個別issue添付の独立したlicense表記は未確認）

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `openevolve_audit.json`。

## Raw records found

3,704 records（単位: program checkpoint JSON snapshot）。同一UUIDのsnapshotをまとめる。metadata/issue attachmentsは別の補助raw。

## Integrity / provenance

Raw provenance coverage: 100.00%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{"0": 325, "1": 119, "10": 1, "2": 64, "3": 29, "4": 29, "5": 5, "6": 9, "7": 5, "8": 1}`。
