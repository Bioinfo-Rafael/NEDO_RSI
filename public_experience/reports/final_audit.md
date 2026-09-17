# Final audit

実SQLiteへのread-only検査結果。元DB・Excelを再生成していない。

| Source | Runs | Nodes | Edges | Root* | Max depth | Parent % | Action % | Result % | Score % | Structure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OpenEvolve | 5 | 587 | 582 | 5 | 16 | 99.15 | 97.10 | 100.00 | 86.03 | tree |
| Evo-MCTS | 5 | 2,101 | 673 | 1,428 | 9 | 32.03 | 100.00 | 100.00 | 61.30 | partial_tree |
| ReST-MCTS | 1,934,674 | 5,015,746 | 3,081,072 | 1,934,674 | 36 | 61.43 | 99.64 | 99.64 | 22.87 | linearized_mcts / linear_reasoning |
| SWE-agent / SWE-smith | 49,915 | 1,508,707 | 1,458,792 | 49,915 | 150 | 96.69 | 100.00 | 96.69 | 0.00 | linear |
| Search Agents | 892 | 18,099 | 11,878 | 6,221 | 6 | 65.63 | 93.49 | 17.69 | 92.92 | partial_search_tree |
| Dream-RSI | 0 | 0 | 0 | 0 | 0 | — | — | — | — | unavailable |
| Tree of Thoughts（追加） | 240 | 13,268 | 11,750 | 1,518 | 8 | 88.56 | 100.00 | 100.00 | 100.00 | beam_search_forest / dfs_search_forest |
| TOTAL | 1,985,731 | 6,558,508 | 4,564,747 | 1,993,761 | 150 | 69.60 | 99.70 | 98.73 | 17.98 | mixed |

*Rootは`parent_id IS NULL`の件数。真のrootに加えて公開情報から親を復元できないnodeを含む。
親関係unknownは **6,724** 件。invalid parent=0は、完全な木を復元できたことを意味しない。

## 検証

- cycles = 0（すべてのedgeでchild.depth=parent.depth+1、run一致）
- duplicate node_id = 0（PRIMARY KEYと全source統合INSERTで確認）
- invalid parents = 0 / orphan events = 0 / foreign key errors = 0
- schema = 元experience.dbと全table/index SQL一致
- source raw modified = 0（全rawのSHA-256照合）
- 元DB・Excel = SHA-256一致、combined内の元全row = 全column一致
- 数値score = nodeごとにraw event上の出典値と一致を全件build時評価。sourceのないscoreはNULL

Raw保管: 4,787 files / 11,006,714,640 bytes（ZIPとその展開物を両方含む）。

## Excel

ユーザー選択により3シートのrun単位サンプル。全件数・収録数・省略数・run ID・長文セルの扱いは[excel_sample.md](excel_sample.md)。DBは全件。

## 不完全なデータ

ReST-MCTSは公開training pathで、full search treeではない。SWE-agentはlinear。Evo-MCTSとSearch Agentsには未復元parentや未公開観測がある。詳細はsource別reportとunavailable_sources.md。

## 取得容量の定義

履歴rawの取得対象は 3,872 files / 8,015,222,920 bytes。ZIP展開前の値で、cloneから取り出した履歴ファイルを含みます。展開物を含む保管数は上記Raw保管の値です。repoコード・.git・生成DBの容量は含みません。詳細: download_totals.json。

## 実行確認

- 実公開recordを用いた14 regression tests: PASS（tests.log）。
- 既存print_trees.pyを最終public_experience.dbへ実行: PASS（print-tree-final.log）。
- Excelは2冊・各3シートの表示を確認。696,448セルを明示した表示用サンプルと照合し一致（excel_audit.json）。日付保護・長文・空白の扱いはexcel_sample.md。
- 公開Excelの選択runと各table収録件数は、最終public DBのSQL結果に一致。
