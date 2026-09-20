# Storage audit

計測日: 2026-09-20。GB/MBはdecimal。SQLite `dbstat` はtableとindexが占めるpage bytes、columnは全行の `sum(length(cast(column AS BLOB)))` によるUTF-8 text bytes。後者はrecord header・数値・index・pageの余白を含まない。サンプル外挿ではなく全行集計。

## DB別の容量と構造

| Source DB | Bytes | Runs | Nodes | Events | Branch points | Branch runs | Branchなしruns |
|---|---:|---:|---:|---:|---:|---:|---:|
| experience_store/experience.db | 20,869,120 | 3 | 23 | 6,873 | 2 | 2 | 1 |
| public_experience/normalized/dreamrsi.db | 49,152 | 0 | 0 | 0 | 0 | 0 | 0 |
| public_experience/normalized/evomcts.db | 5,849,088 | 5 | 2,101 | 2,818 | 85 | 5 | 0 |
| public_experience/normalized/openevolve.db | 240,082,944 | 5 | 587 | 3,933 | 143 | 5 | 0 |
| public_experience/normalized/restmcts.db | 20,147,339,264 | 1,934,674 | 5,015,746 | 5,015,746 | 0 | 0 | 1934674 |
| public_experience/normalized/searchagents.db | 219,025,408 | 892 | 18,099 | 21,800 | 2387 | 694 | 198 |
| public_experience/normalized/sweagent.db | 24,833,101,824 | 49,915 | 1,508,707 | 1,508,707 | 0 | 0 | 49915 |
| public_experience/normalized/treeofthoughts.db | 41,136,128 | 240 | 13,268 | 13,268 | 2709 | 240 | 0 |
| public_experience/public_experience.db | 45,594,738,688 | 1,985,731 | 6,558,508 | 6,566,272 | source別合計と重複 | — | — |
| public_experience/combined_experience.db | 45,615,620,096 | 1,985,734 | 6,558,531 | 6,573,145 | source別合計と重複 | — | — |

## Public DBの物理page容量

| Table / index | Bytes | GB | 比率 |
|---|---:|---:|---:|
| experiences | 27,903,766,528 | 27.9038 | 61.20% |
| raw_events | 12,916,088,832 | 12.9161 | 28.33% |
| runs | 2,115,756,032 | 2.1158 | 4.64% |
| sqlite_autoindex_experiences_1 | 475,521,024 | 0.4755 | 1.04% |
| idx_raw_events_node_id | 451,444,736 | 0.4514 | 0.99% |
| idx_raw_events_run_id | 428,064,768 | 0.4281 | 0.94% |
| idx_experiences_run_id | 427,651,072 | 0.4277 | 0.94% |
| idx_experiences_parent_id | 352,825,344 | 0.3528 | 0.77% |
| sqlite_autoindex_raw_events_1 | 340,365,312 | 0.3404 | 0.75% |
| sqlite_autoindex_runs_1 | 118,796,288 | 0.1188 | 0.26% |
| idx_raw_events_sequence_no | 64,450,560 | 0.0645 | 0.14% |
| sqlite_schema | 4,096 | 0.0000 | 0.00% |

## Source別の主要column（全行のtext bytes）

| Source | Column | Bytes |
|---|---|---:|
| experience.db | raw_events.raw_json | 12,370,797 |
| experience.db | raw_events.stdout | 1,651,079 |
| experience.db | raw_events.source_file | 1,268,817 |
| experience.db | raw_events.command | 720,089 |
| experience.db | raw_events.message | 396,342 |
| experience.db | raw_events.event_id | 206,190 |
| experience.db | experiences.artifacts_json | 147,909 |
| experience.db | raw_events.run_id | 144,060 |
| dreamrsi.db | experiences.artifacts_json | 0 |
| dreamrsi.db | experiences.branch_id | 0 |
| dreamrsi.db | experiences.commit_hash | 0 |
| dreamrsi.db | experiences.commit_hashes_json | 0 |
| dreamrsi.db | experiences.commit_message | 0 |
| dreamrsi.db | experiences.display_path | 0 |
| dreamrsi.db | experiences.git_commit_after | 0 |
| dreamrsi.db | experiences.git_commit_before | 0 |
| evomcts.db | raw_events.raw_json | 1,546,170 |
| evomcts.db | experiences.metadata_json | 1,258,154 |
| evomcts.db | experiences.result_summary | 430,547 |
| evomcts.db | raw_events.source_file | 198,791 |
| evomcts.db | experiences.source_reference | 172,526 |
| evomcts.db | experiences.proposal | 166,210 |
| evomcts.db | experiences.source_files_json | 156,057 |
| evomcts.db | experiences.prompt | 138,550 |
| openevolve.db | raw_events.raw_json | 205,560,326 |
| openevolve.db | experiences.prompt | 9,006,996 |
| openevolve.db | experiences.proposal | 8,961,166 |
| openevolve.db | experiences.result_summary | 8,799,227 |
| openevolve.db | raw_events.source_file | 874,324 |
| openevolve.db | experiences.metadata_json | 622,640 |
| openevolve.db | experiences.display_path | 422,813 |
| openevolve.db | raw_events.node_id | 370,560 |
| restmcts.db | experiences.metadata_json | 2,491,380,206 |
| restmcts.db | raw_events.raw_json | 2,137,810,661 |
| restmcts.db | experiences.prompt | 1,921,376,800 |
| restmcts.db | experiences.result_summary | 1,781,802,928 |
| restmcts.db | runs.metadata_json | 917,637,009 |
| restmcts.db | experiences.proposal | 882,664,243 |
| restmcts.db | experiences.display_path | 561,098,506 |
| restmcts.db | experiences.source_reference | 456,670,998 |
| searchagents.db | raw_events.raw_json | 85,984,849 |
| searchagents.db | experiences.prompt | 45,034,988 |
| searchagents.db | experiences.result_summary | 24,428,931 |
| searchagents.db | experiences.metadata_json | 10,630,803 |
| searchagents.db | experiences.proposal | 9,135,829 |
| searchagents.db | experiences.display_path | 2,339,256 |
| searchagents.db | raw_events.source_file | 1,692,819 |
| searchagents.db | experiences.source_reference | 1,570,821 |
| sweagent.db | raw_events.raw_json | 7,151,880,601 |
| sweagent.db | experiences.prompt | 4,032,999,916 |
| sweagent.db | experiences.result_summary | 3,782,337,158 |
| sweagent.db | experiences.display_path | 2,570,975,004 |
| sweagent.db | experiences.metadata_json | 2,056,894,116 |
| sweagent.db | experiences.proposal | 1,102,815,663 |
| sweagent.db | experiences.source_reference | 138,546,293 |
| sweagent.db | experiences.source_files_json | 127,011,626 |
| treeofthoughts.db | experiences.metadata_json | 8,477,710 |
| treeofthoughts.db | raw_events.raw_json | 5,769,802 |
| treeofthoughts.db | experiences.display_path | 2,205,570 |
| treeofthoughts.db | experiences.result_summary | 1,900,580 |
| treeofthoughts.db | experiences.prompt | 1,502,007 |
| treeofthoughts.db | experiences.proposal | 1,254,579 |
| treeofthoughts.db | experiences.source_reference | 1,170,417 |
| treeofthoughts.db | experiences.source_files_json | 1,093,440 |

## なぜ大きいか・今回の縮小方法

主因はraw_eventsだけではない。public DBはexperiencesが約27.90 GB、raw_eventsが約12.92 GB、runsが約2.12 GB。ReSTは累積prompt/resultと各stepのmetadata、SWEは観測・prompt・raw_jsonに加えて全祖先IDを連結したdisplay_pathの重複も大きい。publicのstdout/message列はNULLで、主にraw_jsonへ内容を格納する。localはraw event本文・投影列が大部分。

元DBの情報を維持して容量を減らす一般案は、raw event全文を外部の圧縮ファイルに置いてhash/locatorで参照すること、固定source metadataを共有化すること、display_pathをparentから必要時に計算すること、累積推論prefixを差分で保存すること。具体的削減量は再設計後の検証が必要で、上のcolumn bytesを単純合算して削減保証とはしない。元DBにはこの変更を加えていない。

branch memoryはlinear-only runを除外し、parentで確認できる分岐点とそのincoming/terminal区間だけを採用する。raw_events tableは複製しない。非branch nodeを1つのownerへだけ割り当て、root anchorはrunsに1回保存する。DB原本に戻れる参照を残す。samplingで選外になる意思決定は新DBには含まれず、情報を全件維持した圧縮とは区別する。

SQLite freelist pages（public）: 0。VACUUM等は一切実行していない。
