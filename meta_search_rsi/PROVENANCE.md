# Provenance

全実装・管理fileの一覧。分類は `VENDORED / ADAPTED / PAPER_INSPIRED / ORIGINAL` の4つだけ。**runtimeにVENDORED/ADAPTEDの著者コードは0件**。依存packageから論文implementationを呼ぶ処理もない。手法をそのまま再現したという意味での「論文準拠」は主張しない。

| Our file | Origin | Paper | Upstream file used | What changed / 責務 |
|---|---|---|---|---|
| `src/meta_search_rsi/__init__.py` | ORIGINAL | — | none | package marker、副作用なし |
| `src/meta_search_rsi/db.py` | ORIGINAL | — | none | 既存SQLiteのread-only SQL adapter |
| `src/meta_search_rsi/tree.py` | ORIGINAL | — | none | Run/Node、実parent、cycle検査、prefix/component |
| `src/meta_search_rsi/atoms.py` | PAPER_INSPIRED | SGA-MCTS §3.3.2 | none | 新規edge abstractionとtyped slot fallback |
| `src/meta_search_rsi/skills.py` | PAPER_INSPIRED | LifeMem §4 / App. L | none | 新規structural workflow fallback |
| `src/meta_search_rsi/retrieval.py` | ORIGINAL | — | none | BM25、metadata filter、証拠多様性 |
| `src/meta_search_rsi/policy.py` | PAPER_INSPIRED | Dream-RSI §3 | none | budget配分concept。全rankingは独自heuristic |
| `src/meta_search_rsi/replay.py` | PAPER_INSPIRED | Dream-RSI §3 / App. B.2 | none | root-only、一般treeへ拡張した逐次replay |
| `src/meta_search_rsi/llm.py` | ORIGINAL | — | none | 任意のCodex CLI呼出を集約、専用home/JSON schema/timeout |
| `src/meta_search_rsi/cli.py` | ORIGINAL | — | none | CLI統合、出力先検査、JSON入出力 |
| `autoresearch/__init__.py` | ORIGINAL | — | none | repository-local harness marker |
| `autoresearch/codex_loop.py` | ORIGINAL | — | none | Codex提案・手動JSON・dry-run・feedback・再開 |
| `autoresearch/prepare.py` | ORIGINAL | — | none | selected run snapshot、task分離、hash固定 |
| `autoresearch/evaluate.py` | ORIGINAL | — | none | 限定AST interpreter、比較・keep/discard |
| `autoresearch/policy_candidate.py` | ORIGINAL | — | none | 実際に改善対象となる選択program |
| `autoresearch/program.md` | ORIGINAL | — | none | 既存autoresearchの運用patternを参考にした新規手順 |
| `tests/test_codex_loop.py` | ORIGINAL | — | none | 模擬CLI、採否/失敗/timeout/再開、auth分離 |
| `tests/test_phase1.py` | ORIGINAL | — | none | 合成fixtureと境界・出典の回帰検証 |
| `configs/default.yaml` | ORIGINAL | — | none | 固定実験設定。JSON-form YAML、外部parser不要 |
| `AGENTS.md` | ORIGINAL | — | none | Codex向けの変更範囲と実行手順 |
| `autoresearch.sh` | ORIGINAL | — | none | 既存CLIへの短い入口 |
| `examples/demo.sh` | ORIGINAL | — | none | 実DBでのbounded end-to-end demo |
| `pyproject.toml` | ORIGINAL | — | none | package metadata、runtime依存なし |
| `.gitignore` | ORIGINAL | — | none | このdirectoryの生成物だけ除外 |
| `outputs/.gitkeep` | ORIGINAL | — | none | 出力directoryのmarker |
| `README.md` | ORIGINAL | 各研究の対応はPAPER_MAP | none | 日本語の用途・操作・検証結果 |
| `ARCHITECTURE.md` | ORIGINAL | WorldEvolverはPhase 2設計のみ | none | データフロー、観測境界、将来interface |
| `PAPER_MAP.md` | ORIGINAL | 4論文 | none | 一次資料の確認と採用/非採用の対応表 |
| `PROVENANCE.md` | ORIGINAL | — | none | この一覧とfile hash manifest |
| `REVIEW_GUIDE.md` | ORIGINAL | — | none | 15分で読む順番 |
| `PAPER_FIDELITY.md` | ORIGINAL | 4論文 | none | 再現の近さ、相違、簡略化理由 |

## 混在fileの関数・class単位の境界

| File | PAPER_INSPIRED | ORIGINAL |
|---|---|---|
| `atoms.py` | `build_atoms` のSGA着想、`delexicalize` のtyped slot着想 | ExperienceAtomのOutcome付きschema、ID、DB field mapping、score/metric差、`outcome_label`、`atom_document` |
| `skills.py` | `build_skills` のworkflow再利用 | Skill container、exact step grouping、IDと出典、`skill_document` |
| `policy.py` | 観測treeに対するbudget配分というmoduleの役割 | `PolicyDecision`、`choose` と全baseline、`decision_for`、`validate_decision` |
| `replay.py` | `replay` のprefix-only reveal | `ReplayResult`、`policy_context` のwhitelist/opaque ID、`attainment`、`summarize`、forest/1child/costへの拡張 |

主要sourceの冒頭にorigin header、混在する関数/classにはdocstringのOriginを記載。`PAPER_INSPIRED` 関数内のDB固有mapping等はこの表でも分けている。upstreamの行・prompt・関数bodyのコピーはない。

## 調査用sourceとlicense

[Paper Map](PAPER_MAP.md) に公式repository、固定commit、読んだfileとsymbolを記録した。LifeMem/WorldEvolverはroot licenseが確認できず、製品コードに取り込まない。Dream-RSIはruntime未公開、SGAは公式repository未特定。

`outputs/research/` の論文HTMLと著者source snapshotは**閲覧・照合のためのlocal参考資料**であり、アプリからimport/実行しない。Git除外しており、配布するthird_party implementationではない。本文textはHTMLからの抽出物。著者source snapshotは無変更で保存し、著者の権利を本プロジェクトのものとは扱わない。コピーしたupstreamを改造したruntimeがないため、`third_party/`、`third_party_patched/`、`PATCHES.md` は作っていない。

## 生成物の追跡

- 全管理fileのSHA-256・origin: `outputs/provenance_manifest.json`。
- 調査資料の取得先/固定commitとSHA-256: `outputs/research/evidence_manifest.json`。
- memory/replay/evaluationは上記ORIGINAL/PAPER_INSPIRED pipelineの生成物。データ自体のsourceは各record内のDB/run/node IDを使う。rawの著作権までORIGINALとする意味ではない。
- trialのcandidate snapshotはORIGINAL。`results.jsonl` にcandidate SHA-256、`manifest.json` に固定file/入力hashを記録。古い開発studyは現実装へのコード変更が検出され、継続不可。最新の検証studyは `outputs/final_study.txt` に示す。


## Codex連携追加

Codex CLI 0.154.0の `exec --help` と [公式non-interactive仕様](https://developers.openai.com/codex/noninteractive)、[公式認証仕様](https://developers.openai.com/codex/auth) を確認。CLIのコードをコピーしたものではなく、subprocessによるORIGINAL adapter。`llm.py` の2関数が唯一のモデル呼出境界で、model名は利用者指定またはCLI既定値。実Codexは今回呼び出さず、fake executableを使って実subprocessと固定評価の接続を検証する。
