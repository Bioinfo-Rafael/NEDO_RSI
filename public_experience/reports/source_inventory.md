# Source inventory

調査日: 2026-09-17。主要6 source + 追加2候補を調査、追加採用はTree of Thoughtsのみ。

| Source | Status | Repo/data | Structure | Nodes | License |
|---|---|---|---|---:|---|
| OpenEvolve | obtained | https://github.com/algorithmicsuperintelligence/openevolve/issues/156 | tree | 587 | Apache-2.0（repository。個別issue添付の独立したlicense表記は未確認） |
| Search Agents | partial | https://github.com/kohjingyu/search-agents#agent-trajectories | partial_search_tree | 18,099 | MIT（repository）。Drive archive内の独立したlicenseは未確認 |
| Evo-MCTS | partial | https://github.com/iphysresearch/evo-mcts | partial_tree | 2,101 | GPL-3.0（repository） |
| ReST-MCTS | partial | https://github.com/THUDM/ReST-MCTS#data--model | linearized_mcts / linear_reasoning | 5,015,746 | PRM-0th: Apache-2.0。他19 datasets: CC-BY-4.0（各HF cardData）。code repositoryのlicenseは未確認 |
| SWE-agent / SWE-smith | obtained | https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories | linear | 1,508,707 | MIT（repository/Hugging Face dataset） |
| Dream-RSI | unavailable | https://github.com/zhengkid/Dream-RSI | unavailable | 0 | 未確認 |
| Tree of Thoughts（追加） | obtained | https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/logs | beam_search_forest / dfs_search_forest | 13,268 | MIT（repository） |

追加候補LATS: https://github.com/lapisrocks/LanguageAgentTreeSearch をcloneし実装・saved logsを調査。全tree edgeが保存されておらず、累積accをnode rewardと混同しないため今回は未採用。上限5 datasetsに対し採用1件。

全repoのcommit SHA、release、file count、HF dataset revision・sizeはinventory.json / additional_inventory.json / hf_inventory.json。

HF各datasetのlicenseはhf_inventory.json.info.cardData.licenseに保存。rawの再配布権を一律にMIT等へ変更していない。
