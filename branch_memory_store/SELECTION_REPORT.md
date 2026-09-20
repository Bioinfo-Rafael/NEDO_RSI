# Selection report

seed=17、最大unit文字数=100,000、有限のsemantic処理枠=100件。

| Source | Total branch points | Selected | Discarded | Reasons |
|---|---:|---:|---:|---|
| nedo_rsi | 2 | 2 | 0 | linear_only: 1 runs |
| dreamrsi | 0 | 0 | 0 | raw recordsなし |
| evomcts | 85 | 25 | 60 | {'stratified_sampling': 60} |
| openevolve | 143 | 25 | 118 | {'input_size_limit': 98, 'stratified_sampling': 20} |
| restmcts | 0 | 0 | 0 | linear_only: 1934674 runs |
| searchagents | 2,387 | 24 | 2,363 | {'input_size_limit': 84, 'stratified_sampling': 2279} |
| sweagent | 0 | 0 | 0 | linear_only: 49915 runs |
| treeofthoughts | 2,709 | 24 | 2,685 | {'stratified_sampling': 2685} |

## 選定の実装

原本はlocal experience.db + source別normalized/*.dbだけ。public/combinedから同じrowを重ねて取り込まない。全eligible NEDO分岐を優先し、その後sourceを均等に巡回する。source内はtask_name hash・depth bucket・branching factor bucket・score有無・status・parent relationで層化し、seed+source_hashのSHA256順で選ぶ。Pythonのランダムhashや現在時刻に依存しない。

OpenEvolve/Evoをsource巡回の先頭に置く。完全tree/明示parentを持つOpenEvolveを含める一方、少数sourceが大量task sourceに埋もれないようsource単位で均等配分する。完全tree優先を全件の厳密な辞書順にした設計ではない。選定は5GBを埋めるためではなく、モデル処理時間を有限にするため。実測input bytesは選定JSONに残し、長大unitを切り詰めずinput_size_limitとして選外にした。これはstorage_capによる除外ではない。

unknown_parentのNULLを架空rootへ結び直さない。確認できる2本以上の子を持つ部分木は、元の不確実性を保持したままeligibleにする。invalid/cycle/duplicateは抽出エラーとし、恣意的に修復しない。

圧縮リンクは元branch point IDを維持する。next/previousが選外のunitを指す場合がある。そのリンクは存在しない新DB rowを実在するように扱わず、source側参照として残す。SQLでLEFT JOINし、未収録を判別する。

## Coverage

全branching runの元node割当はoutputs/coverage.jsonl.gz。最終選定後のownerが選外ならcoverage CLIはdiscarded_samplingを返す。branchなしrunの各nodeはDB/run所属を照会してexcluded_linear_runと判定する。非branch rootはrunsに保存し、coverageではunresolved + reason=root_anchor_in_runsで区別する。branchのない孤立componentはunresolved + component_without_branch_or_missing_parent。

source別全node数・linear-only run数はSTORAGE_AUDIT.md。全branch IDの選定/除外理由はoutputs/selection.json。linear_onlyはrun単位の除外であり、存在しないbranch pointをdiscarded件数へ足さない。
