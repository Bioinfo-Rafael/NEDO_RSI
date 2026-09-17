# 15分のコードレビューガイド

すべて新規実装。`PAPER_INSPIRED` は論文conceptを使う意味で、著者コードの再現性保証ではない。各file冒頭と混在関数のOriginを確認する。

| 順序 | File / 目安 | 何を見るか・由来 |
|---|---|---|
| 1 | [README.md](README.md) / 1分 | 入力・出力・demo・結果。ORIGINALの利用説明。 |
| 2 | [ARCHITECTURE.md](ARCHITECTURE.md) / 1分 | データと観測境界。Phase 2は設計だけ。 |
| 3 | [PAPER_MAP.md](PAPER_MAP.md) / 2分 | 論文version・公式commit・確認したsymbol。upstream使用はNONE。 |
| 4 | [PROVENANCE.md](PROVENANCE.md) / 1分 | 全fileのoriginと、混在関数の境界。 |
| 5 | [tree.py](src/meta_search_rsi/tree.py) / 2分 | `SearchTree`、`prefix`、`component`、`frontier`。ORIGINAL。`db.py::connect` のread-only URIも確認。 |
| 6 | [atoms.py](src/meta_search_rsi/atoms.py) / 1分 | `build_atoms` / `delexicalize`。SGA着想。score/metric差・ID・Outcomeは独自mapping。 |
| 7 | [skills.py](src/meta_search_rsi/skills.py) / 1分 | `build_skills`。LifeMem着想の構造fallback。実action以外のstepを作らない。 |
| 8 | [retrieval.py](src/meta_search_rsi/retrieval.py) / 1分 | `Retriever.search`。ORIGINAL。除外をBM25計算前に適用、成功/失敗/別構造を混ぜる。 |
| 9 | [policy.py](src/meta_search_rsi/policy.py) / 1分 | `choose` / `validate_decision`。Dream-RSIのbudget配分concept、rankingは独自。 |
| 10 | [replay.py](src/meta_search_rsi/replay.py) / 2分 | `policy_context` にhidden情報がないか、`replay` の実edge、評価専用 `attainment`。 |
| 11 | [policy_candidate.py](autoresearch/policy_candidate.py) / 1分 | 唯一の変更対象。`evaluate.py::Candidate` のAST制限と `prepare.py::verify` の固定file検査も読む。 |
| 12 | [PAPER_FIDELITY.md](PAPER_FIDELITY.md) / 1分 | 論文との相違と、demoの限界。 |

最後に `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`。特にhidden score/ID/child数の不変性、同run memory除外、forestのroot-only、candidateのファイルアクセス拒否を確認する。


## Codex連携をレビューする場合

1. READMEの「Codexを使ったautoresearchへの統合」とAGENTS.mdを読む。
2. `autoresearch.sh` → `cli.py`：同じPython CLIへ接続されること。
3. `autoresearch/codex_loop.py::run_loop`：有限回、bestからの再開、手動JSON/dry-run。
4. `src/meta_search_rsi/llm.py::invoke_codex`：唯一のモデル呼出、argv・専用home・timeout。
5. `autoresearch/evaluate.py::evaluate_study`：候補だけを限定ASTで評価、crash保存、best保持。
6. `tests/test_codex_loop.py`：模擬CLIでのkeep/discard/失敗/再開。実モデルを呼ばない。
