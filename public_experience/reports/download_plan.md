# Download plan

取得前調査: 2026-09-17。空き容量 約1 TiB。コードはshallow clone、モデル・Docker・benchmark snapshotは取得しない。

| Source | Repo/dataset | 推定容量 | ファイル数 | tree-native | Public access | 方法 |
|---|---|---:|---:|---|---|---|
| openevolve | https://github.com/algorithmicsuperintelligence/openevolve @ `411fb59c886c18704caaffb611e17cf9e7d824d2` | git報告 7116 KiB; 履歴候補 55,325 B | 候補 22 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| search_agents | https://github.com/kohjingyu/search-agents @ `7c35ac9eb7fda663d821449efdfd44d360fd0e18` | git報告 32433 KiB; 履歴候補 2,298,371 B | 候補 28 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| evo_mcts | https://github.com/iphysresearch/evo-mcts @ `ea70538cc6f3f6da6050947d6296785f56111636` | git報告 6280 KiB; 履歴候補 909,004 B | 候補 8 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| rest_mcts | https://github.com/THUDM/ReST-MCTS @ `2d5f488c3d6e24f99d50a9860e818383b1bb5883` | git報告 3283 KiB; 履歴候補 1,329,678 B | 候補 17 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| swe_agent | https://github.com/SWE-agent/SWE-agent @ `3ea751c087f32b16e039a2233dd6eefecef325d5` | git報告 72169 KiB; 履歴候補 1,910,814 B | 候補 38 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| dream_rsi | https://github.com/zhengkid/Dream-RSI @ `4149ea9181ab1db80f85717ffda2c9f0f130e85b` | git報告 1647 KiB; 履歴候補 0 B | 候補 0 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| myopenevolve | https://github.com/ypwang61/MyOpenEvolve @ `c4a83739900dd5676e343619f48b21436b1d0794` | git報告 4176 KiB; 履歴候補 207,528,824 B | 候補 3914 | OpenEvolve/Evo-MCTSのみ構造要確認 | yes | git clone --depth 1、履歴をrawへcopy |
| Search Agents storage | https://drive.google.com/file/d/127GqJ19qxpAcWlUKXlr5zBeAIW5Pi_0H/view | 未確定、応答ヘッダ確認 | ZIP 1 | 保存実装を確認 | viewは取得済、download未確認 | 公開download URL、必要なら確認form |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS-Llama3-8b-Instruct-PRM-1st | 540,310,781 B | 4 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS-PRM-0th | 271,714,815 B | 3 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-EM-CoT_1st | 65,861,448 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-EM-CoT_2nd | 20,640,910 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-MCTS_Policy_1st | 24,034,975 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-MCTS_Policy_2nd | 22,185,728 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_Self-Rewarding-DPO_1st | 55,504,329 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_Self-Rewarding-DPO_2nd | 14,110,233 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-EM-CoT_1st | 51,957,334 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-EM-CoT_2nd | 26,045,688 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-MCTS_1st | 37,166,756 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-MCTS_2nd | 19,361,812 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_Self-Rewarding-DPO_1st | 30,633,397 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_Self-Rewarding-DPO_2nd | 30,633,397 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-EM-CoT_1st | 41,550,255 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-EM-CoT_2nd | 16,089,822 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-MCTS_Policy_1st | 16,086,691 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-MCTS_Policy_2nd | 21,209,008 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_Self-Rewarding-DPO_1st | 48,067,649 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| ReST-MCTS | https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_Self-Rewarding-DPO_2nd | 15,851,002 B | 1 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |
| SWE-agent | https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories | 4,223,841,784 B | 32 | linear/linearized; native tree未確認 | yes | SHA固定resolve URL |

SWE-smithのtool/xml/ticksは同じ軌跡の別encoding。まずtool全8 shardを取得し、重複trajectoryとstep-prefixの有無を実レコードで確認する。trainはcardにない旧encodingのため同じ軌跡なら重複取得しない。ReST-MCTSはREADMEの全20 datasetのJSONを取得する。

Dream-RSIは現在paper/project pageのみで履歴候補0・release0。公開ページも再確認し、実履歴がなければunavailableとする。

追加sourceは基本6件の処理後、最大5件の範囲で調査する。実データが確認できたもののみ追加する。

Search Agents: Google Driveの公開確認画面でZIP 2.0 GBを確認。trajectory archiveのみ取得する。

## 追加探索（2 sources、最大5以内）

- treeofthoughts: https://github.com/princeton-nlp/tree-of-thought-llm @8050e67d0e3a0fddc424d7fa5801538722a4c4cc; 履歴候補 12 files / 19,614,847 bytes。tree-nativeかは実コード照合。公開git shallow clone、モデル等はなし。
- lats: https://github.com/lapisrocks/LanguageAgentTreeSearch @853d81614607dd27433faf17c7b0a7d660f95d22; 履歴候補 0 files / 0 bytes。tree-nativeかは実コード照合。公開git shallow clone、モデル等はなし。

SWE-smith再確認: tool 24,100 rows と公開cardのxml/ticks件数が一致しないため、異なるtrajectoryの欠落を防ぐため残るxml/ticks/trainも全shard取得する。既知の総容量4,223,845,920 B、全32 parquet。traj_idで重複を避け、異なるencodingのrawは保存する。

OpenEvolve issue #88: https://github.com/user-attachments/files/20842742/f170432e-eb4b-4e94-b045-c3745efacc49.json; 1補助checkpoint/log、容量未確定。公開添付を取得、履歴との対応が確定しなければunbound raw eventとして保存。

OpenEvolve issue #141: https://github.com/user-attachments/files/21194986/openevolve_20250712_015411.log; 1補助checkpoint/log、容量未確定。公開添付を取得、履歴との対応が確定しなければunbound raw eventとして保存。
