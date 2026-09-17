# Architecture

```text
Existing SQLite DB (read-only)
        |
        v
    SearchTree
        |
        +------> ExperienceAtom ----+
        |                           |
        +------> Skill -------------+--> Retriever
                                        |
                                        v
                               Meta Search Policy
                                        |
                                        v
                               Historical Replay
                                        |
                                        v
                               Replay Evaluation
                                        |
                                        v
                                  autoresearch
```

## Phase 1の境界

- `db.py`: SQLとread-only接続だけ。`tree.py` がレコードから木を組み立てる。
- memory用runと評価runを分ける。memoryのsource run/node/DBを保持する。評価runと同じrunのmemoryは使わない。同taskの別runも評価では除外する。
- replay内部だけが全treeを持つ。policyには観測済みnodeを新しいopaque IDで渡し、run metadata・元のsequence_index・未観測child数は渡さない。
- 子が残っているかを先読みしてfrontierを絞らない。空probeを実行して初めて枯渇を知る。1要求=1budget。固定順で1つの実childを公開する。
- `DEEPEN` / `WIDEN` / `REVISIT` はbudgetの使い方を表す。保存済みtreeでの遷移は全て「選択した観測済み親から次の未観測childを1つ公開」。未知actionは作らない。
- 数値は同run・同score名・同方向だけ比較する。全treeから計算するrankは評価器だけが使い、policyには渡さない。
- policy_candidateは限定Python subsetで、JSONの観測のみを受け取る。import・属性アクセス・ファイル操作を許可しない。固定file/入力snapshotのSHA-256で評価の前後を検査する。
- 原DBは変更しない。生成物は `outputs/`。全件memory構築を暗黙に行わず、run IDを明示して読む。

## Phase 2（設計のみ、未実装）

WorldEvolverに着想を得た将来interface:

```text
predict_outcome(current_state, proposed_action, retrieved_memories)
    -> predicted_outcome, uncertainty, evidence
```

予測を使うときも「観測された事実」と「予測」を別型・別出力にする。今回のreplayには予測値を入れない。将来はMeta Search Policyが `SIMULATOR / WORLD_MODEL / POLICY / DATA_EVALUATOR` の対象を選び、Task Agentが具体的改変を担当する。Simulator/World Modelの自動生成・co-evolution・NN学習は未実装。

## データ・境界の詳細

- 3つの入力DBを確認済み: `../experience_store/experience.db`、`../public_experience/public_experience.db`、`../public_experience/combined_experience.db`。共通3-table schema。接続は `mode=ro` と `query_only=ON`。`raw_events` は今回ロードせず、既存ノードのprompt/proposal/resultを使う。
- `Run` / `ExperienceNode` はDBのsubset。全raw contextをpolicyへ丸ごと渡さない。`source_run` と `source_nodes` が元DBのIDへ戻る鍵。
- `SearchTree.prefix` は実親を含む閉じた部分木を作る。`frontier()` は観測root＋観測leaf。replayでの候補集合は、一般treeの内部分岐にも戻れるよう「全観測nodeから、空probe済みだけ除外」に拡張する。
- **forestのreplayは1componentずつ**。既定は記録順で最初のroot、単独replayの `--root-id` で変更可能。最初はそのrootだけ観測。component外node数を報告し、rankの分母もそのcomponent内に限定する。欠落parentから始まるcomponentは真の初期状態とは限らない。
- child順は `sequence_index, node_id`。環境がこの順で返すが、policyが見られるのはreveal順のopaque ID。同時実行・未知action・合流DAGは扱わない。
- contextの文字列は既定800字、memoryはnodeごと最大6件の原文/抽象文を各800字に制限。budgetも有限。選択runの上限は既定5,000nodeで、超過時は切り捨てずerror。全件memory用のstreaming indexは今後の拡張。
- memoryは独立のJSON生成物でDBへ逆書込しない。atomは実edgeのみ。skillは実root→leafのみ。policyはretrieveしたactionを実行せず、success/failure evidenceを選択点の加点/減点として使う。
- `plateau_rounds` は最後の観測edgeで同cohortの改善がなかった連続回数。異なるmetricを比較しない。raw metric deltaも同run同key内の差で、unit互換は推測しない。
- score正常化には全componentのscoreが必要なので `attainment` は評価器側専用。policyの `observed_rank` は現在見えている同cohortの値だけから作る。

## 信頼範囲

候補はPython ASTの限定subsetとして解釈し、関数call自体を認めない。ファイル・ネットワーク・import・属性・外部globalへの経路はない。入力はJSONコピー、step燃料とtimeoutあり。固定source/inputsのハッシュ検査も併用する。候補の実行に対する境界であり、OS利用者によるmanifestと評価器の同時改竄を防ぐものではない。

観測済みの元ログ本文自体に未来の集計が書かれている場合、それを意味的に自動判別するLLMはない。metadataや全run集計は渡さないが、データに混入した情報を完全に消せるとは主張しない。研究評価前にはsourceごとの時間整合性と重複問題を監査する。


## Codexを開発者役として接続する

```text
採用policy + 契約 + 集約feedback
           ↓
Codex CLI（llm.py、read-only、専用CODEX_HOME）
           ↓ hypothesis/code JSON
codex_loop.py → 固定evaluate.py → replay.py
           ↓
       keep / discard / crash
           ↓
採用版だけ更新 → 次の有限iteration
```

現在のCodex sessionや別のautoresearch driverからは、同じJSONを `--proposal-file` へ渡せる。候補生成は差し替え可能でも、評価器の入力・指標・baselineは固定。runnerはrootのseedを書き換えず、study内のcode snapshotを実行対象にする。

Codexが出すPythonは直接execしない。既存の限定ASTを通してpolicyとして解釈する。CODEX_HOMEはこのdirectoryのoutputs/codex-homeで、global configを読み込まず、file保存のChatGPT認証を使う。CLIでread-only sandboxを要求するが、ローカル管理者の強制設定による挙動まで独自OS隔離で保証するものではない。

改善役には集約したdevelopment評価feedbackを渡す。raw DB全体・secretはpromptに埋め込まない。policy実行時のprefix-onlyと、開発時にfeedbackを見ることを区別する。モデル呼出中のeventとstderrはiteration内に保存し、timeoutやCtrl-C時は起動したprocess groupを停止する。

二重loopと同時評価はそれぞれfile lockで拒否。構文エラーもtrialとして残す。採用snapshotにhashを持たせ、再開時に確認する。固定sourceに変更があれば新studyを要求し、旧manifestは維持する。
