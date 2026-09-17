# 取得できない・復元できない情報

調査日: 2026-09-17。取得済みrawを根拠に記録し、欠落をsynthetic dataで補完していません。

| Source | 確認した場所 | 状況・不足 | Code onlyか |
|---|---|---|---|
| Dream-RSI | https://github.com/zhengkid/Dream-RSI / releases / https://www.dream-rsi.com/ | 実discovery trees・replay worlds・traces・checkpointsの公開ファイルが見つからない。0 run | 論文・assets・release planのみ。codeも準備中 |
| Search Agents | READMEの公開Drive ZIP、run.py / RenderHelper | VisualWebArenaの910 HTMLを取得。18は空。別WebArena archiveのリンク、未選択branchの観測、曖昧なdeep candidateのparentは確認できない | codeと部分的な実履歴あり |
| Evo-MCTS | paper_data JSONL / 全5 production logs / serialization実装 | 指定JSONLは取得成功したがparent/childrenを保存していない。logの明示Father Objから一意に復元できないedgeはNULL | codeと部分的な実履歴あり |
| ReST-MCTS | README Data & Modelの20 HF datasets / MCTS実装 | 全25 data filesを取得。公開policy/PRM pathsから元の完全MCTS treeのnode identity/全branchは復元できない | codeとlinearized pathsあり |
| LATS（追加候補、未採用） | https://github.com/lapisrocks/LanguageAgentTreeSearch / programming/mcts.py / saved JSONL | implementations/reflections/feedbackは見つかったが全parent graphではない。accは累積task精度なのでnode rewardにしない | codeと一部logsあり |

OpenEvolve/SWE-agent/Tree of Thoughtsを含む採用sourceの既知のデータリンクは取得できました。上記はネットワーク障害を意味するものではなく、公開範囲・保存形式の制約です。取得URL・SHA・サイズはinventory / download manifestに残しています。
