# Paper / Code Mapping

確認日: 2026-09-17。実装前に作成し、実装後に差分を追記する。論文本体・付録と公開repositoryを確認した。すべて新規コードであり、論文の再現実装とは主張しない。

## Dream-RSI

- Paper: [Dream-RSI: Recursive Self-Improvement through Evolving Worlds, v1](https://arxiv.org/html/2609.14858v1)
- Supplementary: 同論文 Appendix B.2（prefix-only policy improvement prompt）、本文 §3 のreplay仕様。
- Official repository: [zhengkid/Dream-RSI](https://github.com/zhengkid/Dream-RSI)
- Pinned commit: `4149ea9181ab1db80f85717ffda2c9f0f130e85b`
- 公開状態: README / papers / assets。実行コードは準備中。**paper description only**。
- 今回利用するconcept: historical discovery tree、prefix-only replay、探索budget配分、固定replay評価を用いたpolicy program改善。
- 今回利用するupstream code: **NONE**。付録のprompt/codeもコピーしない。
- 新規実装: `replay.py` の逐次reveal、`policy.py` の選択、`autoresearch/evaluate.py` の採否。
- 差分: 論文のroot＋linear branches・並列batchから、一般tree/forest・逐次1要求へ拡張。内部nodeにも戻れる。数値rewardを混ぜずrun内rankで集計。結果がないprobeにも1budgetを使う。online再配備は実装しない。

## SGA-MCTS

- Paper: [arXiv v1](https://arxiv.org/html/2604.14712v1) / [ACL Findings 2026 正式掲載](https://aclanthology.org/2026.findings-acl.60/)
- Supplementary: 論文 Appendix A（設定）、B（case）、C（tool schemaとworkflow）。ACLページのchecklistも確認対象。
- Official repository: **unavailable / 今回確認した論文・ACLページ・GitHub検索から著者公式repositoryを特定できず**。
- Pinned commit: N/A。**paper description only**。
- 非公式 `msbel5/openclaw-sga-mcts-atoms` は採用しない。
- 今回利用するconcept: §3.3.2 のState–Goal–Action atom、typed slotへの選択的抽象化、§3.4 のmemoryをsoft hintとして使う考え方。
- 今回利用するupstream code: **NONE**。
- 新規実装: `atoms.py`。既存edgeをState–Goal–Action–Outcomeへ写像、原文保持、設定で明示したentity置換。score deltaは同run・同score名・同方向のみ。
- 差分: MCTSによる経験生成、成功経路のみの蒸留、LLMによるschema推定、semantic embeddingを省略。失敗も残す。Outcomeは本システムの拡張。数値を一律maskせず制御値を保つ。

## LifeMem

- Paper: [LifeMem: Enabling Lifelong Experience Reuse for LLM Agents, v1](https://arxiv.org/html/2609.12655v1)
- Supplementary: 同論文 Appendix L（クラスタ割当・skill抽出prompt）、D（retrieval）、I（cluster解析）。本文 §4.1–4.3。
- Official repository: [BITHLP/LifeMem](https://github.com/BITHLP/LifeMem)
- Pinned commit: `b045fccd3d9c3d3e77e17eb5f9fb111dd2886762`
- 調査した該当実装（**参照のみ、コピー・importなし**）:
  - `memory/build_memory.py::modify_clusters` とstage 1/3：LLMによるNEW/ADD、insight構築。
  - `memory/cluster_sqlite.py::ClusterDB.split_cluster` / `_check_cluster_cohesion_mrl` / `update_db`：embedding MRLで分割判定。
  - `memory/cluster_sqlite.py::ClusterDB.add_insights`：最大50 trajectoryを選びrule抽出。
  - `memory/insight_extraction.py::my_main` / `get_insight` / `RuleManager.parse_and_apply_updates`：LLMによるrule追加・更新・削除。
  - `memory/edit_memory.py::retrieve_exemplar_name`、`memory/server.py::retrieve`：FAISS exemplar検索とLLM選択。
- 今回利用するupstream code: **NONE**。repository rootのlicenseを確認できないためコードを取り込まない。同名 `halsayxi/LifeMem` は別研究。
- 新規実装: `skills.py`。実在するroot→leaf経路をworkflowとして保存し、同じ抽象step列だけをまとめる。
- 差分: LLM意味クラスタリング・MRL分割・rule蒸留は実装しない。成功/失敗patternは元のstatus/結果または同run score差に根拠がある場合のみ。構造fallbackは論文のsemantic skillと同等ではない。

## WorldEvolver

- Paper: [Self-Evolving World Models for LLM Agent Planning, v2](https://arxiv.org/html/2606.30639v2)（v1も確認）
- Supplementary: 本文 §3.2 とAppendix A–C（設定・実装・評価）。
- Official repository: [magicgh/WorldEvolver](https://github.com/magicgh/WorldEvolver)。v2本文のcode availabilityから確認。
- Pinned commit: `f7013c63a0b012d3ba1f0fbbefab9218ba4d110e`
- 調査した該当実装（参照のみ）:
  - `src/world_model/episodic_semantic.py::WMEpisodicSemantic.predict` / `update`：memory付き予測と観測後更新。
  - `src/world_model/modules/memory_mixins.py::EpisodicMemoryMixin._append_episodic_transition` / `SemanticMemoryMixin._update_semantic_memory`。
  - `src/world_model/modules/state_library.py::EpisodicLibrary.retrieve_top_k`：actionに基づく検索。
  - `src/world_model/modules/selective_foresight.py::SelectiveForesightGate.apply`：confidenceによる出力抑制。
- root licenseは確認できず、コードはコピー・importしない。
- 今回利用するconcept: 実transitionを根拠とするmemory、predictionの不確実性とevidenceを分離する将来設計。
- 今回利用するupstream code: **NONE**。
- 今回の実装: **なし**。`ARCHITECTURE.md` のPhase 2 interfaceだけ。予測器・mismatch critic・selective foresightをPhase 1に混ぜない。

## 調査証跡とoriginの境界

取得した論文HTML/本文、GitHub commit/tree/API情報と参照したLifeMem sourceは `outputs/research/` に調査用として保存。アプリは読み込まない。再配布用vendorではない。上記commit URLと論文versionで再確認できる。公開状態は確認時点のもの。

- `VENDORED`: upstreamを無変更で製品に同梱。今回は0件。
- `ADAPTED`: upstream implementationの改変。今回は0件。
- `PAPER_INSPIRED`: 論文のconceptに基づく新規実装。
- `ORIGINAL`: DB接続・境界検証・評価用rank・CLIなど今回固有の設計。

既存 `autoresearch/program.md`、`autoresearch/csv_experiment/run.py`、`public_experience/program.md`、`public_experience/src/autoresearch.py` を読み、「固定データ/評価器、変更可能file限定、候補保存→評価→keep/discard→記録」という運用を採用。ソースコードはコピーしない。無期限実験は行わず、今回は小規模demoを実行する。


## Codex autoresearch連携の追加

論文conceptとsourceの採用範囲は上記のまま。`autoresearch/codex_loop.py` と `src/meta_search_rsi/llm.py` をORIGINALとして追加した。Codex CLIの構造化提案を固定replayへ渡す接続であり、upstream論文codeの追加利用はない。現sessionからの手動JSON提案にも対応し、optionalなモデル呼出はllm.pyへ隔離している。
