# Parser autoresearch 試行記録

既存 `autoresearch/README.md` と `autoresearch/csv_experiment/run.py` を確認し、候補コードの保存・固定evaluator・反復計測の方式を `src/autoresearch.py` に適用しました。元の学習・実験runnerは実行していません。Codexがparserを改良し、runnerが公開rawに対してofflineで検証します。

## 評価条件

- schema/common/evaluatorのSHA-256をfixed_hashes.jsonで固定。rawは別の全件SHA-256 manifestで検証。
- run_idのSHA256 modulo 5でdevelopment/held-outへ分割。問題が重複するreleaseもあるため、未見問題への汎化性能の評価ではありません。
- hard failure: cycle、duplicate ID、invalid parent、sourceにないscore。いずれも0が必須。
- objective = action率 + result率 + provenance率 + known-parent-relation率（0〜4）。実験rewardではありません。
- objectiveだけで採否を決めません。欠落を正しくNULLにして全branchを保持することを優先します。

## 試行結果

| Trial | Source | Runs | Dev nodes | Dev objective | Held-out nodes | Held-out objective | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| 01-baseline | openevolve | 5 | 403 | 3.0000 | 184 | 3.0000 | 0 |
| 01-baseline | evomcts | 5 | 976 | 3.0000 | 1125 | 3.0000 | 0 |
| 01-baseline | restmcts | 1000 | 788 | 4.0000 | 212 | 4.0000 | 0 |
| 01-baseline | sweagent | 18 | 175 | 3.9200 | 30 | 3.9667 | 0 |
| 01-baseline | searchagents | 0 | 0 | — | 0 | — | 0 |
| 01-baseline | dreamrsi | 0 | 0 | — | 0 | — | 0 |
| 02-responses-and-formats | openevolve | 5 | 403 | 3.9628 | 184 | 3.9891 | 0 |
| 02-responses-and-formats | evomcts | 5 | 976 | 3.0000 | 1125 | 3.0000 | 0 |
| 02-responses-and-formats | restmcts | 1000 | 2666 | 4.0000 | 717 | 4.0000 | 0 |
| 02-responses-and-formats | sweagent | 1000 | 20443 | 3.9615 | 5528 | 3.9622 | 0 |
| 02-responses-and-formats | searchagents | 0 | 0 | — | 0 | — | 0 |
| 02-responses-and-formats | dreamrsi | 0 | 0 | — | 0 | — | 0 |
| 03-evo-parent-evidence | evomcts | 5 | 976 | 3.3227 | 1125 | 3.3600 | 0 |
| 04-search-selected-baseline | searchagents | 60 | 196 | 3.7602 | 58 | 3.7759 | 0 |
| 05-search-all-candidates | searchagents | 60 | 1151 | 2.8523 | 323 | 2.8142 | 0 |
| 06-tot-prefix-audit | treeofthoughts | 240 | 10401 | 4.0000 | 2867 | 3.9951 | 0 |
| 07-search-task-results | searchagents | 892 | 14675 | 2.8149 | 3424 | 2.8277 | 0 |
| 08-all-encodings-final | sweagent | 1000 | 20443 | 3.9615 | 5528 | 3.9622 | 0 |
| 09-swe-list-message-support | sweagent | 1000 | 20443 | 3.9615 | 5528 | 3.9622 | 0 |

## 改良内容と採用理由

1. **OpenEvolve**: initial baselineはcode/metrics/explicit parentsを保持。保存済みprompts.responsesの読取りを追加し570 nodeでactionを保持。新しいactionは生成していません。
2. **ReST-MCTS**:単一pathから明示Step境界への分割、JSONL/columnar DPO/SciGLM variants対応へ改良。中間stepへ終端labelを配布せず、元reasoningが連結で完全一致することをtestしました。
3. **Evo-MCTS**:parent未復元baselineから、official codeの5桁丸めとFather Obj/Depthに基づく一意の対応を採用。673 edgesを復元し、曖昧なparentはNULL。
4. **Search Agents**:選択trajectoryだけのbaseline（最初に展開されたreddit 60 runs）から保存された全candidateへ拡張。同じ60 runsで254→1,474 nodes。未選択branchの観測が公開されていないためresult率とobjectiveは下がりますが、branchを捨てず採用しました。その後ZIP全件展開を完了し892 runsへ拡大、task PASS/FAILをrunとunbound-to-node eventに追加。60 run試行と892 run試行は異なる母集団です。
5. **SWE-agent**:18 demonstrationから公開SWE-smith全parquetへ拡張。4 encodingsのtraj_idを比較しmetadata conflict=0。102,078 rowsを49,897 unique trajectoriesへまとめ、tool→xml→ticks→trainの順で採用。trainのmessagesだけJSON文字列ではなくlistのため対応を追加。最後の2試行は先頭1,000 runsの回帰確認なので同じ数値です。4形式の実データを独立testし、最終buildは全runを評価します。
6. **Tree of Thoughts**:追加sourceとして公式prefix生成/DFS stack実装からparserを作り240 runsを検証。ambiguous prefixはNULL。追加sourceは独立した初期監査で、複数候補比較は行っていません。
7. **Dream-RSI**:実履歴未公開のため変換改善対象は0件。synthetic dataを補っていません。

## 最終検証

- adapter候補スナップショットと各試行JSON: `autoresearch/<trial>/`。
- 全件のdevelopment/held-out結果: `<source>_audit.json`。試行runnerの先頭1,000 run制限を最終DBへ適用していません。
- regression tests: `tests.log`。実公開record、全ReSTファイル形式、SWE4形式、explicit parents、reasoning無損失、score捏造/不正parent/cycle/duplicate拒否、offline実行を確認。
- SQLite統合後の独立検証: `final_audit.md` / `final_audit.json`。

## Held-outの解釈

run単位の分割と別集計は実施しています。ただし全形式の調査・回帰fixtureの参照や、候補ごとの両splitの集計も行っているため、完全に未参照のblind holdoutではありません。形式ごとの変換と構造の監査として解釈してください。
