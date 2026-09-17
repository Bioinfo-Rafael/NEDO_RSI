# Excelサンプルの範囲

source/dataset/original_splitごとに先頭3 complete runs。OpenEvolve/Evo-MCTSは全run。Search Agentsはsiteごとに3 runs。

既存localデータはcombined Excelに全件収録。public DBは全件、Excelはrun単位サンプル。

| Table | Public全件 | Excel収録 | 省略 | Combined全件 | Excel収録 |
|---|---:|---:|---:|---:|---:|
| runs | 1,985,731 | 109 | 1,985,622 | 1,985,734 | 112 |
| experiences | 6,558,508 | 3,773 | 6,554,735 | 6,558,531 | 3,796 |
| raw_events | 6,566,272 | 7,644 | 6,558,628 | 6,573,145 | 14,517 |

32767 UTF-16 code unitsを超えるtextはDB参照へ置換。値の省略はxlsxだけ。DBとrawは全文保存。

長文セル置換数: public=2753, combined=2796。対象IDはexcel_sample_manifest.json。

source_datasetはmetadata_jsonから表示用に展開。DB schemaは変更していない。

ISO日時にはExcelの自動変換・小数秒丸めを防ぐため、文字列保護用の先頭 `'` を付けています。DBの日時は元の文字列のままです。XLSXから読む場合はこの先頭1文字だけを除いてください。XMLで使えない制御文字はExcel表示だけで除去します。

NULLと空文字は、Excelではともに空白セルとして表示します。区別が必要な処理にはDBを使用してください。
