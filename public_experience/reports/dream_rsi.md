# Dream-RSI

Status: **unavailable**

## 取得内容

zhengkid/Dream-RSIのcurrent tree・README・releases・project pageを再確認。SHA 4149ea9181ab1db80f85717ffda2c9f0f130e85b。

## Raw format / tree

repositoryにはpaper/assets/README/release planのみ。実discovery history/replay worlds/checkpoints/trajectory dataは見つからない。

## 共通schemaへのmapping

adapterは0 recordsを返す。demoや論文の図を実行履歴に見立てない。

## 実装根拠

reports/inventory/dream_rsi/{README.md,tree.json,releases.json}; https://www.dream-rsi.com/

## 変換結果

| 指標 | 値 |
|---|---:|
| Runs | 0 |
| Nodes | 0 |
| Edges | 0 |
| parent=NULL | 0 |
| Max stored depth | 0 |
| 2 children以上のnode | 0 |
| 最大children数 | 0 |
| Raw events | 0 |
| Parent coverage (edges/nodes) | —% |
| Action coverage | —% |
| Result coverage | —% |
| Score coverage | —% |
| Unknown parent relations | 0 |

## 復元不能項目・制約

codeもrelease準備中。公開実履歴の代わりにsynthetic dataを作成しない。

## License

未確認

## Development / held-out

run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。

詳細: `dreamrsi_audit.json`。

## Raw records found

0 records（単位: published trajectory）。公開実履歴なし。

## Integrity / provenance

Raw provenance coverage: —%。
cycles=0, duplicate IDs=0, invalid parents=0, unsupported scores=0。統合後のorphan/FK検査はfinal_audit.md。

children数ごとのnode件数: `{}`。
