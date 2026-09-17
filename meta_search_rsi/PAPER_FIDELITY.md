# Paper Fidelity Audit

コードのoriginは `VENDORED / ADAPTED / PAPER_INSPIRED / ORIGINAL` の4種類だけを使う。ここでのFidelityは再現の近さを表し、origin分類とは別。全論文・公式実装の確認先は [PAPER_MAP.md](PAPER_MAP.md)。

| Component | Paper | Fidelity | Exact paper behaviorとして採る部分 | Deviations | 理由 |
|---|---|---|---|---|---|
| Historical replay | Dream-RSI §3, App. B.2 | conceptual（観測境界は近い） | root-onlyで開始し、選択後に過去の観測を返す。policy codeを固定世界で評価 | batchなし、内部nodeも再選択、1child/要求、空probeも課金、forestはcomponent単位 | 既存DBが一般tree/partial/linearであり、並列コスト情報が揃わない |
| Experience atom | SGA-MCTS §3.3.2 | conceptual | State/Goal/Actionと原文を分け、明示entityをtyped slot化 | Outcome追加、成功だけでなく失敗も保存、LLM/schema推定/新MCTSなし | 存在するedgeを監査可能な形で再利用しAPI不要にする |
| Skill abstraction | LifeMem §4, App. L | conceptual / structural fallback | 軌跡と抽象workflowを対応づけ、sourceを維持 | exact step列によるrun内grouping。LLM意味クラスタ・MRL分割・rule更新なし | 欠落actionを補完せず、決定的で読みやすい最小実装にする |
| Retrieval | SGA/LifeMemの着想 | ORIGINALの検索方式 | memoryを補助的証拠として返す | BM25＋metadata filter＋成功/失敗/構造多様性。embedding/LLM rerankなし | API/Vector DBなしの検証用。hybrid semantic検索の再現ではない |
| Meta Policy | Dream-RSIのbudget配分concept | conceptual＋ORIGINAL heuristic | policy programの更新、基盤agent/評価器の固定 | 4種類の配分labelと独自ranking、task action生成なし | 異種履歴に共通のmetaレイヤーだけを検証 |
| Replay metric | — | ORIGINAL | — | run/component内rank・range正規化・rawを分離、cost項0.01、並列bonusなし | 異task rewardの単位/方向の混同を避ける |
| Candidate execution | — | ORIGINAL | — | 限定AST interpreter、固定fileとsnapshot hash | candidateから評価器やDBへアクセスさせない |
| Future outcome prediction | WorldEvolver §3.2, App. A–C | architecture only | 実transitionの根拠、uncertainty/evidenceの分離を参考にする | 予測・semantic rule・selective foresightは一切未実装 | Phase 2の責務だから |

## 公式実装を読んで確認した差

- LifeMemはLLMがworkflowを割当・要約し、embeddingのMRLに基づいてクラスタ分割、別のLLM処理でruleを更新する。こちらのskillは生の経路の構造fallbackであり、その機能や精度を代替したとは主張しない。
- WorldEvolverの `WMEpisodicSemantic.predict/update` は実際にLLM予測と観測後memory更新を行う。今回はそれを呼び出さない。公式 `SelectiveForesightGate.apply` はlogprobがない場合に通過する分岐も持つため、将来uncertainty設計を単に「常に低信頼を遮断」とみなさない。
- Dream-RSIは公式repositoryにruntime codeがない。SGA-MCTSは著者公式repositoryを特定できなかった。この2つは **paper description only**。非公式codeを公式扱いしていない。

## 検証が示す範囲

- 構造・出典・観測境界・read-only・決定性をunit testと実DB demoで確認した。
- hidden nodeのscore/ID/子数を変更しても、reveal前のpolicy inputは同一。
- memory用はlocal 3 run、評価用は公開6 run。同task_nameと同runを除外してから検索する。
- 評価6 run中、数値比較できるのはOpenEvolveの5 run。SWE-smithはscoreを捏造せずNULLとする。
- 同一development集合で候補を改善した結果であり、blind holdout・未知課題汎化・online実験の改善は検証していない。
- memoryあり/なしの結果が同じ場合、そのまま報告する。memory retrievalが実行できることと、有効性があることを分ける。


## Codex proposal loop（追加）

`autoresearch/codex_loop.py` と `llm.py` はORIGINALの接続処理。Dream-RSIの「評価feedbackからpolicy programを改善する」流れを具体化するが、論文の開発agent/promptをコピーしたものではない。提案はCodexのJSON、採点は既存の固定replay。LLMはatom/skill抽象化やreplay内部へ追加していない。今回の連携検証はfake CLI・手動proposal・dry-runであり、実Codexが研究性能を改善したという実験ではない。
