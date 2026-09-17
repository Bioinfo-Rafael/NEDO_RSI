# Public history parser autoresearch

目的は取得済み公開履歴の変換精度改善。モデル学習・LLM API呼び出し・raw生成は行わない。
既存 `autoresearch/csv_experiment/run.py` の「候補コード保存→固定データ評価→採否→記録」を
`src/autoresearch.py` に移した。提案・コード変更はCodexが行い、runner自体はLLMを呼ばない。

- 変更可能: `src/adapters/*.py` のみ。
- 固定: `raw/`、`src/schema.sql`、`src/common.py`、`src/audit.py`、期待する不変条件。
- まずrawと公式の生成・保存処理を読む。source ID・parent・scoreを推測で作らない。
- developmentで仮説→変更→評価を行い、変更コードと結果を保存する。
- cycle / duplicate ID / invalid parent / scoreの出典不一致は即reject。
- source recordの保持、公式根拠によるedge、action/result/provenanceの保持を優先する。
- `objective` はcoverage率の診断値。未選択branchを新たに取り込むと観測率が下がり得る。
  その場合は保持件数と欠落理由を記録し、率を上げるためにbranchを捨てない。
- scoreがないsourceはNULLのまま。SWE-smithのboolean resolvedを数値rewardへ変換しない。
- 全ソースのfinal auditではrun_idのSHA256 modulo 5によるdevelopment/held-outを別集計する。
  同じ問題の異なる公開pathを完全に独立な問題集合とみなさない。これはparserの構造検証でありモデルの一般化評価ではない。

```bash
.venv/bin/python src/autoresearch.py --label unique-candidate-name --source openevolve
.venv/bin/python -m unittest discover -s tests -v
```

評価履歴は `reports/autoresearch/`。候補が不採用ならそのadapterだけ保存済み候補へ戻す。
既存の研究コード、local Experience Store、git履歴には手を加えない。

## Held-outの解釈

run単位の分割と別集計は実施しています。ただし全形式の調査・回帰fixtureの参照や、候補ごとの両splitの集計も行っているため、完全に未参照のblind holdoutではありません。形式ごとの変換と構造の監査として解釈してください。
