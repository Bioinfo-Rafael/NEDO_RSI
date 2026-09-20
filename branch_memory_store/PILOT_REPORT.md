# Pilotとsemantic処理の検証

実Codex CLIを使用した測定。外部provider SDKや固定の疑似LLM応答は成果物生成に使っていない。unit testの模擬応答とは別。

| Pilot | Input units | Successful | Failed | JSON parse | Elapsed seconds | Mean input chars | Mean output chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 10 | 10 | 0 | 10 | 184.67 | 31391.9 | 2576.7 |
| v2 | 3 | 3 | 0 | 3 | 65.16 | 57679.0 | 2872.3 |

## Distribution / token usage

- v1: sources={'nedo_rsi': 2, 'openevolve': 2, 'evomcts': 2, 'searchagents': 2, 'treeofthoughts': 2}、平均output bytes=2580.7、CLI usage={'input_tokens': 234004, 'cached_input_tokens': 10880, 'output_tokens': 6544, 'reasoning_output_tokens': 1118}。
- v2: sources={'nedo_rsi': 2, 'openevolve': 1}、平均output bytes=2872.3333333333335、CLI usage={'input_tokens': 93563, 'cached_input_tokens': 0, 'output_tokens': 2207, 'reasoning_output_tokens': 583}。

input_tokensはCLIが報告した値で、raw unitだけでなくCodexのsystem/developer context等も含む。cached_input_tokensは内数として別掲する。reasoning tokenをoutputと二重加算しない。

v1の逐次throughputは 0.0542 units/s。単純外挿では100件が約 30.8 分、全5,326件が約 27.3 時間。実際は文字量・共有context cache・並列度・混雑で変わる。4並列の理想値は約1/4だが保証しない。巨大unitは選定外なので全件外挿は特に粗い。

金額はnot available。利用者の契約・利用枠に依存し、CLIログに課金額はないため推測しない。全件処理を開始せず、最終100件に限定した。

## 人が確認した内容

v1では10/10が構造検証を通ったが、OpenEvolveの出力に「26-circle packing」等のtask固有表現が残った。v2で具体的名称・数・benchmarkを機能表現へ置換する指示を追加。旧cacheを新promptの結果として採用しない。

v2のlocal2件とOpenEvolve1件をrawと照合した。localではobjectiveの定義がraw.evidenceにあり、terminalとbaselineの改善方向、keep/discardの区別が支持される。OpenEvolveではbranch-pointのtimeoutとterminalのvalid/combined-scoreを照合し、baselineにmatching scoreがないためoutcome_type=unknownを維持している。

追加でEvo-MCTSのopaque operator e3、Search Agentsの異なる目標を述べるcandidate、ToTのincomingとnext-only unitを確認した。operatorの意味を捏造せずunknownにすること、candidate valueを実行成功へ言い換えないこと、未収録continuationの結果を補完しないことを確認。全100件の文ごとの完全な意味監査ではない。

## 品質検査の範囲

全件にschema、node ID/順序、terminalのstep数、continuation、決定論的outcome typeの検査を適用。数値はrawの正本に保持する。自由文の含意や抽象度は自動検証だけで証明できず、rawとの対比とuncertaintiesを利用者が確認する。

## 最終v2 datasetの処理実績

v2 pilot 3件 + 本処理97件 = 100件completed。failed=0。本処理は4並列、537.01秒（0.1806 units/s）。本処理開始時のcache hitは3件。

本処理のattempts=98、JSON parse成功=98。1応答がtrajectory step数検証で拒否され、再試行で成功した。v2全体では101呼出、最終採用100件。v1 pilotの10呼出は別で、最終DBへ混入しない。

採用100件の平均input=29897.31 chars、平均output=2939.18 chars。採用応答のCLI usage合計={'input_tokens': 2287315, 'cached_input_tokens': 870400, 'output_tokens': 74203, 'reasoning_output_tokens': 16466}。このusageは拒否された応答およびv1 pilotを含まないため、全試行の請求token合計とは区別する。

本処理後の再実行は100 cache hit、0 attempts、追加モデル呼出0。記録はoutputs/resume_check.json。全件のraw/schema/outcome検査、2table整合性、圧縮全5,326点のcycle/親子整合性を検証した。
