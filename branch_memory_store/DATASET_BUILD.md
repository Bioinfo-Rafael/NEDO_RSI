# 既存Experience DBの構築仕様

調査開始: 2026-09-20。対象は既存のローカルファイルと実装。既存DB・raw・adapterを読み取り専用で確認し、このdirectory内だけに調査結果を保存する。以下は現在の変換仕様であり、公開元の最新仕様を推測したものではない。

## 再構築の境界

`public_experience/src/registry.py` に7 adapterが登録されている。`common.node` / `common.event` / `common.finish` が3-table canonical recordを作り、`build_db.build` がsource別DBへ挿入する。`combine` がpublic DBを作り、`combine_existing_public` が既存local DBを加えてcombined DBを作る。Dream-RSI adapterは空で、公開実行履歴を生成しない。

公開データの再取得は取得時commit・HF revisionを固定する必要がある。既存 `download.clone` は現在HEADのshallow clone後に記録SHAとの一致を要求するため、将来そのまま実行すると失敗し得る。再現時は先に記録SHAをcheckoutする。inventoryの再調査は最新SHAへ変わるため、取得済みinventoryを保存して使う。

local DBには非公開のローカルautoresearch履歴とCodex rolloutが含まれる。GitHub cloneだけでは復元できない。`experience_store/ingest_manifest.json` が指定するraw、`_ingest_raw/git/WorldModel_plants.git`、元のrun directoryが別途必要。未公開の入力を公開URLで取得可能とは扱わない。

## 共通field mapping

`h(x) = sha256(str(x).encode()).hexdigest()[:20]`。public ID生成の文字列化はPython `str`、tupleの場合はそのreprであり、JSON hashではない。

| 入力 / 処理 | canonical field | 変換・NULL条件 |
|---|---|---|
| adapterで定義するrid | runs.run_id / experiences.run_id | source別規則は後述 |
| original | experiences.node_id | `rid + '::' + original` |
| parent | experiences.parent_id | `rid + '::' + parent`、parentがNoneならNULL |
| proposal / prompt / result | proposal / prompt / result_summary | strを維持、dict/listはcompact JSON、None/空文字/空dict/空listはNULL |
| score | score | bool・非数・非finiteはNULL、ほかfloat変換 |
| metrics | metrics_json | 未指定なら`{}`。scoreとは別に原値を保持 |
| original,parent,record,source + adapter metadata | metadata_json | original_node_id / original_parent_id / source_record等。finishがdataset/structure/parent_relationとstate/action/next_stateのcolumn参照を追加 |
| path,record | source_reference / source_files_json | `public_experience`からの相対pathと`#record=N`、file配列 |
| 検証済みparent chain | depth / display_path | edge数とrootからのnode ID列。重複node・cycle・不明parentはbuildエラー |
| nodeへのevent集合 | event_count / first_event_sequence / last_event_sequence | 件数・min/max。eventなしは0/NULL/NULL |
| roots / dataset / metadata | runs | rootが1つだけならroot_node_id、ほかNULL。project_name=dataset、task_name=metadata.task_nameまたはdataset。status=recorded。nodeあり・root1つ・incompleteでなければreplay_eligible=1 |
| event(source,rid,nid,path,record,payload,type,sequence) | raw_events | event_id=`source::event::h((rid,nid,relative_path,record,sequence,type))`。raw_json=payloadのcompact JSON。source_lineは物理lineとは限らずrecord番号。binding_method=source_record、confidence=1.0 |

public adapterが明示しない列はschema.sqlのdefaultまたはNULL。特にrunのstart/end/git_branchはNULL、thread/session配列は`[]`。experienceのbranch_id/attempt_index/commit/trace日時はNULL、artifact/commit配列は`[]`。public raw eventのstdout/stderr/message/role/tool/command等はNULLで、内容はraw_jsonへ保存される。

## 調査と再現資料

調査中に既存のdownload/build/export scriptは実行しない。全sourceの実ファイル名、SHA-256、取得revisionは `source_manifest.json`（取得時コピーはoutputs/evidence.json）、raw sampleのformat/field名は `outputs/raw_samples.json`、実DBのsqlite_master/PRAGMA/行数は `outputs/audit.json` に保存する。

## Sourceごとの取得・変換

以下のcommitは既存inventoryの取得時記録。tag/releaseは全sourceで使用していない。取得時刻の厳密なwall timeはnot recorded（既存報告の調査日は2026-09-17）。raw file一覧は `outputs/evidence.json` のraw_filesを正とする。

### OpenEvolve

- Source: OpenEvolve、circle packingの実行履歴。公式 https://github.com/algorithmicsuperintelligence/openevolve (`411fb59c886c18704caaffb611e17cf9e7d824d2`)。
- 実データ: https://github.com/ypwang61/MyOpenEvolve (`c4a83739900dd5676e343619f48b21436b1d0794`)。公式issue #156から取得先を発見。
- raw: `raw/openevolve/repository/**/programs/*.json`。JSON Program checkpoint。同UUIDをまとめ、`prompts`のJSON文字数が最大のsnapshotを採用。すべてのsnapshotはeventに残す。
- 補助raw: issue #88 https://github.com/user-attachments/files/20842742/f170432e-eb4b-4e94-b045-c3745efacc49.json と #141 https://github.com/user-attachments/files/21194986/openevolve_20250712_015411.log 。runを確定できずunbound event。
- converter: `public_experience/src/adapters/openevolve.py:OpenEvolve.bundles/unbound_events`。
- parent: **explicit**。元`parent_id`のみ。欠落parentを補わずエラー。

| Raw field | Canonical field | Transformation / NULL |
|---|---|---|
| parentを辿ったroot UUID | runs.run_id | `openevolve::MyOpenEvolve::{root UUID}` |
| id / parent_id | node_id / parent_id | 共通prefix規則 |
| generation, iteration_found, id | sequence_index | この順でsortした0始まりindex |
| metadata.changes_description / prompts.*.responses[] | proposal | 前者のtruthy値を優先、なければ非空responsesのJSON。どちらもなければNULL |
| 親のcode / metrics | prompt | parent_code / parent_metricsをJSON化。rootはNULL |
| code | result_summary | 子program全文 |
| metrics.combined_score | score / score_name / score_direction | keyがあればname=combined_score、maximize。数値無効ならscore=NULL、keyがなければname=NULL / unknown |
| metrics | metrics_json | 全metrics保持 |
| generation, iteration_found, timestamp, prompts位置 | metadata_json | snapshot_count、parent_relation=explicit、score_evidenceのevent ID/pathも付与 |
| checkpoint全体 | raw_events.raw_json | program_checkpoint、snapshotごとに1行。補助JSONはissue_checkpoint_without_run、補助logは行ごとissue_log_without_run（run/node NULL） |

### Evo-MCTS

- Source/repository: https://github.com/iphysresearch/evo-mcts (`ea70538cc6f3f6da6050947d6296785f56111636`)。
- raw: `raw/evo_mcts/repository/execution_logs/evo-mcts/**/merged_log.log`、run1の`breakthrough_nodes/*.json`と`results/paper_data/mcts_tree_nodes_pt5_algorithm.jsonl`。
- converter: `adapters/evo_mcts.py:EvoMCTS.bundles`。
- parent: **reconstructed_from_official_code / unknown**。公式evolution_interfaceの5桁丸めを参照。Action行より前の同operator/depth/objectiveの子が一意、さらに子より前のdepth-1/father objectiveが一意の場合だけedgeを作る。根拠行と丸め桁をmetadataに保持。depth1でも実rootは欠落しforestとなる。

| Raw field | Canonical field | Transformation / NULL |
|---|---|---|
| merged_log親directory名 | runs.run_id | `evomcts::{directory name}` |
| Eval_times | original/node_id | 数字文字列を共通prefix化、数値順にsequence_index |
| ActionのFather Obj / Now Obj / Depth | parent_id | 上の一意性検査。不確定はNULL |
| Operator | proposal | 通常operator文字列。paper recordがあればoperator/thinking/reflection/algorithmのJSON |
| breakthrough system_content/user_content/*_reflection | prompt | 存在するkeyのみJSON、通常NULL |
| Objective value / paper code / breakthrough code | result_summary | 通常objective JSON、paper codeで置換、breakthrough codeがtruthyならさらに置換 |
| Objective value | score/name/direction | 有限数のみfloat、objective / minimize。無効数はscore=NULL / status=evaluation_failed |
| paper objective, fitness | metrics_json | paper recordがある場合のみ2値、通常`{}` |
| parsed evaluation, paper, breakthrough, accepted Action evidence | raw_events | evaluation / paper_node / breakthrough_record / accepted_expansion。score_evidenceはevaluation.objective |
| 原depth, iteration, evidence | metadata_json | original_depth、parent_evidence、missing_structural_root、paper位置。run incomplete=True |

### ReST-MCTS

- Repository: https://github.com/THUDM/ReST-MCTS (`2d5f488c3d6e24f99d50a9860e818383b1bb5883`)。
- Dataset: 公式READMEに記載された https://huggingface.co/zd21 配下の20 dataset。**全dataset名・revision・各URL・SHA-256は `outputs/evidence.json` のdownloadsに記録**。HF `resolve/{revision}/{file}`で取得。
- raw: `raw/rest_mcts/zd21__*/*.json`。JSON配列、拡張子.jsonのJSONL、columnar DPO object。`common.json_records/columnar`で形式を分岐。配列長不一致はエラー。
- converter: `adapters/rest_mcts.py:RestMCTS.parse_record`。
- parent: **linear_sequence**。明示された`Step N:`または`步骤 N:`だけで分割。chosen/rejectedは別run。共通prefixからbranchを作らない。

| Raw field | Canonical field | Transformation / NULL |
|---|---|---|
| relative file path / record index / choice | runs.run_id | `restmcts::h(relative_path)::{1-based index}-{choice}` |
| step index | node_id / parent_id | original=1,2,…、親=直前step、最初NULL |
| instruction → prompt → content | 初期context / runs.task_name | `dict.get`のfallback順。present-but-NULLは次fieldへfallbackしない |
| prompt_answer | context + solution | 最初の`\nSolution:\n`で分割。separatorなしはcontext=NULL、全文solution。choice=prefix |
| output / summary / chosen+rejected / response_chosen+response_rejected | solution / choice | 順に優先。output/summaryはpolicy、対はchosenとrejected |
| step文字列 | proposal | 空ならNULL |
| context + 過去step prefix | prompt | 結合した文字列、空ならNULL |
| 過去prefix + 今step | result_summary | 累積推論、空ならNULL |
| prompt_answerのlabel | score / score_name | 最後stepに限りscore=float(label)、name=label。directionは**unknown**。他形式・途中stepはNULL |
| 未指定 | metrics_json / status | `{}` / NULL |
| choice, step_index/count, record_index, split | metadata_json | datasetは親directory名の最初の`__`を`/`へ。run structureは名前に基づきlinearized_mctsまたはlinear_reasoning |
| record全体 / step | raw_events.raw_json | 最後は`{"record":d}`、途中はstepと位置。reasoning_record / reasoning_step |

### SWE-agent / SWE-smith

- Repository: https://github.com/SWE-agent/SWE-agent (`3ea751c087f32b16e039a2233dd6eefecef325d5`)。
- Dataset: https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories 。取得時revisionはevidenceのdownloadsに固定。
- raw: `raw/swe_agent/repository/trajectories/demonstrations/**/*.traj` と `raw/swe_agent/SWE-bench__SWE-smith-trajectories/data/*.parquet`（全32 shard）。
- converter: `adapters/swe_agent.py:demonstrations/smith/parse_messages`。
- parent: **linear_sequence**。1 assistant message = 1 action turn。SWE-smithの別encodingはtool→xml→ticks→train、同encoding内filename順で最初のtraj_idだけ採用。raw encoding全件を捨てないがcanonical nodeは重複させない。

| Raw field | Canonical field | Transformation / NULL |
|---|---|---|
| .traj relative path / traj_id | run_id | `sweagent::demonstration::h(path)` / `sweagent::SWE-smith::{traj_id}` |
| turn index | node_id / parent_id | 0始まりturn、直前turnが親。metadata.original_node_idはSmithでは元message index |
| trajectory[].action | proposal（demo） | 原値。prompt=前turn.observation、result=今turn.observation |
| message.tool_calls または content | proposal（Smith） | tool_callsがtruthyなら優先 |
| 初回messages[:assistant index] / 2回目以降は直前assistant後〜今回前 | prompt | context/直近tool観測のJSON。全prefixを毎回保存しない |
| 今assistant後〜次assistant前のmessages | result_summary | 観測JSON。空配列ならNULL |
| resolved | metrics_json / status | 最後turnだけ`{"resolved":value}`とresolved/unresolved。途中metrics=`{}`、status=NULL |
| 未指定 | score/name/direction | NULL/NULL/unknown。resolvedを数値rewardへ変換しない |
| assistant, observations, 初回context, 最後outcome | raw_events.raw_json | 1 turn1 event。最後outcomeはmessages以外の元record |
| thought / instance_id / index / prefix参照 | metadata_json | runにはmodel・encoding・resolved・traj_id。demoはinfo/demonstration |

### Search Agents

- Repository: https://github.com/kohjingyu/search-agents (`7c35ac9eb7fda663d821449efdfd44d360fd0e18`)。
- 公開archive: https://drive.google.com/file/d/127GqJ19qxpAcWlUKXlr5zBeAIW5Pi_0H/view 。Drive revisionは**not recorded**。ダウンロードZIPのSHA-256で同一性を確認する。
- raw: `raw/search_agents/extracted/**/render_*.html` と同directoryのresults.txt。ZIPは別途保持。
- converter: `adapters/search_agents.py:pages/SearchAgents.bundles`。
- parent: **reconstructed_from_official_code / linear_sequence / unknown**。公式HTML生成のNew Page区切りを使う。depth0 candidateは直前実行nodeへ、深いcandidateは同batchで同a_idxかつdepth-1が一意ならそこへ。未知はNULL。選択actionの一意一致時は既存candidateを再利用し、なければexecuted nodeを追加。

| Raw field | Canonical field | Transformation / NULL |
|---|---|---|
| parent directory / HTML stem | run_id | `searchagents::{directory}::{render stem}` |
| 初回観測 / batch b,candidate j / executed page i | original ID | initial_observation / b{b}c{j} / executed{i} |
| New Page前text | task_name / initial.prompt | HTMLからtext抽出 |
| state_obv pre | prompt / selected result_summary | 当該観測をprompt、次page観測を結果。最終結果はNULL |
| additional_textのcandidate prediction / Selected action / predict_action | proposal | evaluated candidateと選択actionを区別 |
| candidate score | score/name/direction | 有限float / value / maximize。初回・fallback実行nodeはNULL |
| 未指定 | metrics_json / status | `{}` / NULL。PASS/FAILをcandidate scoreにしない |
| a_idx,curr_a_idx,depth,batch,selected | metadata_json | 復元根拠、run incomplete/unsaved_branch_observations=True |
| parsed page / candidate / results.txt行 | raw_events | rendered_step / evaluated_candidate / task_result。task_resultはrunありnode=NULL、複数attemptを保持し対応を捏造しない |

### Tree of Thoughts

- Repository/data: https://github.com/princeton-nlp/tree-of-thought-llm (`8050e67d0e3a0fddc424d7fa5801538722a4c4cc`)。
- raw: `raw/treeofthoughts/logs/**/*greedy*.json` と `logs/crosswords/infoss_dfs*.json` の4 search filesを変換。取得した他8 JSONはbaseline/cacheで変換対象外。
- converter: `adapters/treeofthoughts.py:beam/dfs`。
- parent: **reconstructed_from_official_code / unknown**。beamは直前levelのcandidateで`step.ys`に含まれる一意prefix。DFSは`actions[:-1]`に一致する記録が一意のときだけ親。prompt-only rootを人工追加しない。

| Raw field | Canonical field | Transformation / NULL |
|---|---|---|
| file / idxまたはrecord index | run_id | `treeofthoughts::h(relative_path)::{idx}`、DFSは1始まりrecord index |
| step index + candidate index / DFS record index | original ID | beam=`{si}-{j}`、DFS=`{i}` |
| step.x + 一意parent prefix | prompt（beam） | question/prefix JSON。不確定prefixはNULL |
| new_ys[j] | proposal / result_summary | 一意prefix差分 / 全文。parent不明ならproposalも全文 |
| values[j] | score/name/direction/metrics | value / maximize / `{"value":...}`。alignment不一致はエラー |
| actions[:-1], actions[-1], info | prompt/proposal/result（DFS） | action_stack JSON / 最後action / info JSON |
| info.r_word | score/name/direction | r_word / maximize。metrics_json=info全体 |
| step, candidate_index, select_new_ys, actions, env_step,total_step | metadata_json | selected/原depth/implicit_root等。DFS metadata.original_node_idはtotal_step |
| beam候補のx/value/prefix / DFS record全文 | raw_events | beam_candidate / dfs_expansion、score_evidenceに元path |
| forest | runs.metadata_json | incomplete=True、missing_prompt_root=True。status等は共通default |

### Dream-RSI / その他

https://github.com/zhengkid/Dream-RSI (`4149ea9181ab1db80f85717ffda2c9f0f130e85b`) は取得時paper/project/release planのみ。`DreamRSI.bundles()`は空、mapping対象rawとDB rowは0。後日公開されたかどうかをこの調査では主張しない。

LATS (`lapisrocks/LanguageAgentTreeSearch`、`853d81614607dd27433faf17c7b0a7d660f95d22`) は調査済みだが現在DB未収録。sourceを追加したことにはしない。

### NEDO_RSI local（Rafael / Nulla1202）

`dream_rsi_experience_store/src/experience_store/discover.py` → `ingest.py` → `classification.py` → `bindings.py` / `commit_bindings.py` → `scripts/simplify_store.py` が履歴から確認できる変換経路。

**再現上の不整合:** 現在の`src/experience_store/schema.py`は現3-tableの`dream_rsi_experience_store/schema.sql`を読むだけだが、残っているingest.pyは旧`sources` / `discovery_runs`等を要求する。旧DDLは現在のこの2 directoryには存在しない。したがって現在のingest_all.pyを実行してlocal DBをゼロから再構築できるとは記載しない。既存local DBの読み取り専用snapshotを再構築入力として渡すか、旧DDLと当時の全raw/refsを別途回収する必要がある。これはpublic7 adapterの再現可能性とは別の制約。

`stable_id(prefix,*parts)` = prefix + `_` + SHA256(`\x1f`.join(str(parts)))先頭24桁。publicの20桁hashとは別。

| Raw / legacy field | canonical field | 規則 |
|---|---|---|
| research/runs/{run}/manifest.json | runs.run_id | stable_id(run,rafael,run directory name)。config/model/start/end等をsession/runへ |
| experiments/*/experiment.json | experiences.node_id | stable_id(node,run_id,exp.idまたはdirectory名) |
| baseline / arm / round | parent_id | baseline後、同arm直前node（confidence .8）、なければbaseline（.5）。**inferred_chronological**であり明示parentではない |
| source/hypothesis.json.hypothesis | proposal | なければNULL |
| experiment.objective | score/name | 数値ならobjective、後段classificationでソースコード根拠にminimize。非数はNULL |
| metrics全numeric leaf | metrics_json | legacy node_metric_links JOIN metrics。nameごと重複(value,unit,direction)を除き、複数値はarray |
| decision / observation_text | result_summary | observation_text優先、なければdecision文字列、なければNULL |
| prompt_text / error / valid / completed | prompt / status | legacy値。fail/error/valid=0ならfailed、completed/evaluated/valid=1/commitならcompleted、それ以外unknown |
| Nulla作者commitとproduction thread | run_id / node_id / parent_id | stable_id(run,nulla,control/production-20260915b)、stable_id(node,run_id,commit)。既知Git親を優先(.95)、なければ直前commit(.4) |
| Codex JSONL / file manifest | raw_events | event ID=stable_id(event,file_id,line_no)。parserがtool/message等を投影、raw_jsonも保存。redactを通す |
| node_event_bindings | raw_events.node_id | confidence降順、同点はexplicit>filesystem>generation_id>git>temporal順。未割当はNULL。保持runにないevent.run_idもNULL |
| node_artifact_links / node_commit_links | artifacts_json / commit列 | path/type/hash等の重複排除、git_commit_afterまたは最新linked commitをprimaryにする |
| parent_node_id chain | depth/display_path | 実際のlegacy parentのみを辿る。cycle/missing parentはエラー |
| run metadata.run_dir / branch | source_type/reference | run_dirありautoresearch_run、なければgit_codex_reconstruction。nodeはworkspaceありautoresearch_experiment、なければgit_commit |

`simplify_store.convert`はnodeのあるrunだけ保持し、すべてのeventを残す。発見manifestが示すローカルraw・Git branch snapshot・ZIPは第三者へ別途受け渡す必要がある。元absolute pathはsource_referenceやmetadataに入るため、別pathへの移植ではその列が変わる。byte-identical SQLite再生成はSQLite版・挿入順・page layoutにも依存し、ここでは同じrawから同じ論理recordを再生成する手順を示す。

## 再生成command（新しいworkspaceでのみ実行）

既存workspace内のbuild scriptは出力先を上書きするため、このタスクでは実行しない。再現用のコピーを `branch_memory_store/outputs/rebuild/` に作ってから実行する。

```bash
# NEDO_RSIのrootから。prepareはコード・固定manifestだけを新directoryへ配置する。
PYTHONDONTWRITEBYTECODE=1 python3 branch_memory_store/rebuild_sources.py prepare
PYTHONDONTWRITEBYTECODE=1 python3 branch_memory_store/rebuild_sources.py download

cd branch_memory_store/outputs/rebuild
python3 -m venv .venv
.venv/bin/pip install -r public_experience/requirements.txt
.venv/bin/python public_experience/src/prepare_raw.py
cd ../../..
PYTHONDONTWRITEBYTECODE=1 python3 branch_memory_store/rebuild_sources.py verify-raw
cd branch_memory_store/outputs/rebuild
for source in openevolve evomcts restmcts sweagent searchagents dreamrsi treeofthoughts; do
  .venv/bin/python public_experience/src/build_db.py --source "$source"
done
# 合流前に別途入手したlocal experience.dbを、この新workspaceのexperience_storeへ配置する。
.venv/bin/python public_experience/src/build_db.py --combine
.venv/bin/python public_experience/src/prepare_excel.py
# @oai/artifact-toolが利用可能なNode環境で（このpackageはnpm公開packageとは限らない）
node public_experience/src/export_excel.mjs public_experience
node public_experience/src/export_excel.mjs combined_experience
.venv/bin/python public_experience/src/validate_excel.py
```

Excel exporterにはCodex同梱 `@oai/artifact-tool` が必要。入手経路・版が保存されていない他環境でのbyte-identical XLSX再生成は保証できない。表示用データはprepare_excel.pyのJSONから独立に再現できる。Excelはsource/dataset/split単位のrun sampleで、巨大DB全件ではない。source別3 run、OpenEvolve/Evo全run、Search Agentsはsite別3 run。32767 UTF-16単位を超えるセルはDB参照＋先頭1000文字。source_datasetは表示用にmetadataから展開する。

local DBをrawから再作成するには、前述の**欠落している旧DDLを先に復元することが必要**。新workspaceに旧manifestと全入力を用意し、`discover`を再実行せず固定manifestを使用する。以下は旧DDLを回収した後のAPI順序で、現checkoutでそのまま実行できるcommandではない。`ingest_all.py`はdefault出力を削除するため既存workspaceでは使わない。

```python
from pathlib import Path
import sys
root = Path('branch_memory_store/outputs/rebuild').resolve()
sys.path.insert(0, str(root / 'dream_rsi_experience_store/src'))
from experience_store.ingest import ingest
from experience_store.classification import classify_scores_and_runs
from experience_store.bindings import bind_node_traces
from experience_store.commit_bindings import bind_node_commits
con = ingest(root, root/'experience_store/ingest_manifest.json', root/'experience_store/legacy.db')
classify_scores_and_runs(con)
bind_node_traces(con)
bind_node_commits(con)
con.commit()
con.close()
# その後、新workspaceのsimplify_store.py --source-db legacy.db --output-db experience.db
```

raw入力を移す際はmanifest.original_path/archived_copy_pathの存在とhashを照合する。元file_idを再採番しない。元WorldModelのGit refsが変わると作者commit集合も変わる。取得時refsとmanifest.gitを固定し、差分があれば再現成功としない。

## 取得済みHF datasetの全revision

| Dataset | Revision |
|---|---|
| https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories | 08e109b4a59eaeebf80e4675cd125d42e7ac99a4 |
| https://huggingface.co/datasets/zd21/ReST-MCTS-Llama3-8b-Instruct-PRM-1st | 94eb8649c399ce57090d65749ac7affc67a92719 |
| https://huggingface.co/datasets/zd21/ReST-MCTS-PRM-0th | 4318a214504f641641e5e35bb0471a8ff0544295 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-EM-CoT_1st | e6f575bc15dd1e3e4e89055d7a0b347e035c6353 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-EM-CoT_2nd | 80239655d307e9c04d3170774f293643dbb3f83f |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-MCTS_Policy_1st | df0e56ab5d6584a75522b122d6bc59f7b9e1ae7d |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_ReST-MCTS_Policy_2nd | cd41836e87f03a4cd7009e9da6894bf7a1bddf10 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_Self-Rewarding-DPO_1st | 5aabf7b405fe561fbd385c673c24f72edecf4d09 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Llama3-8b-Instruct_Self-Rewarding-DPO_2nd | cf2b9dadc33709095bcb61711dc78669b2b5676f |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-EM-CoT_1st | 0366b4c05ba105d37b09981b7f88b47f198436fe |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-EM-CoT_2nd | b7d81acfc0d47d94037b7e76559a17b0c1bf7080 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-MCTS_1st | 6e47b2fa78e2799ab7859a407adca1081731410c |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_ReST-MCTS_2nd | 4f7fcc3acd8efc391d264e1e0dad23d2d2f093d8 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_Self-Rewarding-DPO_1st | 398a673b8ceaea965285d52d98d3b151c191d458 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_Mistral-MetaMATH-7b-Instruct_Self-Rewarding-DPO_2nd | d97b11b19a7f2901974c097866bb72ad2f9c1004 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-EM-CoT_1st | 0119d196a7926998b1aef6d75896547bb9a3313f |
| https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-EM-CoT_2nd | 48c8fdb41a9a96e0608cb74fa594483ea2e5ed18 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-MCTS_Policy_1st | 95c2103912c33c2dffce015dbf6a3b448f9cdc8f |
| https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_ReST-MCTS_Policy_2nd | 953ce4fc2b04d6fc0f4887d28cb509f206b84494 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_Self-Rewarding-DPO_1st | 98b4b2ad9708838bd83ac4ba4c7806abbeda0ba2 |
| https://huggingface.co/datasets/zd21/ReST-MCTS_SciGLM-6B_Self-Rewarding-DPO_2nd | 3851fda926813e8b847e4ad45f0fa18ecc6d78bd |

## 再計測したcanonical source件数

| Source DB | Bytes | Runs | Nodes | Events | Branch points | Branch runs |
|---|---:|---:|---:|---:|---:|---:|
| experience_store/experience.db | 20,869,120 | 3 | 23 | 6,873 | 2 | 2 |
| public_experience/normalized/dreamrsi.db | 49,152 | 0 | 0 | 0 | 0 | 0 |
| public_experience/normalized/evomcts.db | 5,849,088 | 5 | 2,101 | 2,818 | 85 | 5 |
| public_experience/normalized/openevolve.db | 240,082,944 | 5 | 587 | 3,933 | 143 | 5 |
| public_experience/normalized/restmcts.db | 20,147,339,264 | 1,934,674 | 5,015,746 | 5,015,746 | 0 | 0 |
| public_experience/normalized/searchagents.db | 219,025,408 | 892 | 18,099 | 21,800 | 2,387 | 694 |
| public_experience/normalized/sweagent.db | 24,833,101,824 | 49,915 | 1,508,707 | 1,508,707 | 0 | 0 |
| public_experience/normalized/treeofthoughts.db | 41,136,128 | 240 | 13,268 | 13,268 | 2,709 | 240 |
