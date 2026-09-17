# Policy program autoresearch

## 目的と許可範囲

Origin: ORIGINAL。既存NEDO_RSIのautoresearch運用を参考にした新規実装。
変更してよいfileは **`autoresearch/policy_candidate.py` だけ**。DB、memory、replay、metric、baseline、config、tests、prepare/evaluateは固定。NN学習・provider APIの直接呼出・Simulator生成なし。任意の候補生成はCodex CLIへ委譲する。

1. `examples/demo.sh` またはCLIの `evaluate --prepare-only` でstudyを作る。
2. 初期candidateを `evaluate --study outputs/autoresearch/<study> --label v1` で評価。
3. replay結果から仮説を立て、candidateだけ変更する。
4. 別labelで同studyを評価する。改善時だけ `best_policy.py` が更新される。
5. `discard` なら保存されたbestをcandidateへ戻す（Git resetは使わない）。

評価器は元DBのselected runを `cases.json` に固定して使う。元DB全体を複製しない。memory用runと評価用runは分離し、同task_nameのmemoryも除外する。別表記の同一問題・別runにコピーされたtrajectoryを完全に検出する仕組みはない。本demoはdevelopment評価でありblind holdoutではない。

## 候補言語

`def choose(ctx):` を1つだけ書く。ヘッダーcommentと関数docstringは可。

許可: local変数への代入、`if/elif/else`、JSON配列への有限 `for`、`return`、添字、dict/list/tuple literal、条件式、数値四則演算・剰余、比較、論理式。
禁止: import、関数呼出、属性アクセス、while、再帰、関数追加、contextへの代入、ファイル/ネットワークアクセス。`exec` / `eval` は使わずASTを解釈する。100,000 AST stepと設定秒数で打ち切る。

返却例:

```python
return {'action': 'WIDEN', 'target_node_id': node['node_id'], 'reason': 'plateau'}
```

`ctx` の主な情報:

- `nodes`: 観測済みnode（opaque ID、観測済み親、depth、観測済みchildren_count、scoreと方向、観測内rank、改善の符号履歴、短い結果）
- `frontier`: 空probeで枯渇と確認されていない観測済みnode。内部nodeも含む。
- `memories`: atom/skillの出典・関連度・短い原文/抽象文。各nodeの `memory_support` は成功/失敗根拠の符号付き平均。
- `budget_remaining`, `budget_total`, `round`, `plateau_rounds`, `current_node_id`, `goal`, `seed`

未観測nodeのscore/ID/子数、元DBへの接続や全treeは含まれない。候補はtask agentではなくbudget配分器。

## 固定評価

- 1要求で1つの保存済みchildをreveal。空probeも1budget。
- rank = `(bestのdistinct score順位 - 1) / (distinct score数 - 1)`。minimizeは符号を反転。1値だけなら観測時1、比較可能な値を未観測なら0。
- objective = `rank - rank_cost × used_budget / budget`。設定の既定 `rank_cost=0.01`。
- raw_best・range正規化attainment・rank AUCは別々に記録。raw scoreを異task間で平均しない。
- 比較可能なscoreがないrun、またはscore名/方向のcohortが混在するrunはscalar集計から除外し、件数を報告。
- forestは1つの実componentだけをrootからreplayする。既定は記録順の最初のroot。他componentを接続しない。単独replayでは `--root-id` で選べる。
- 原DBのsize/mtime、固定file集合のSHA-256、snapshot/memory/configのSHA-256を検査。固定file変更後は新studyを作る。既存manifestを書き換えて通さない。

この保護は、限定candidateが評価器へアクセスできないことと改変検出を保証する。OS所有者が評価器とmanifestを同時に書き換える行為を防ぐファイルシステムsandboxではない。

結果: `outputs/autoresearch/<study>/trials/<label>/result.json`、候補snapshot、`results.jsonl`、`best_policy.py`。同点はdiscard。自動採用されるのはbestの保存先であり、candidateの編集内容やGit履歴は自動的に巻き戻さない。


## Codexとの接続契約

`./autoresearch.sh` は `python -m meta_search_rsi.cli autoresearch` の短いwrapper。
通常のpolicy探索で変えられるのは候補programだけ。生成側は `hypothesis` と `code` の2つの文字列をJSONとして返す。hostがstudy内の候補snapshotへ保存して評価するため、seedの `autoresearch/policy_candidate.py` を自動更新する必要はない。

- `--dry-run`: 初期seedのローカル評価＋prompt/argv/schema保存。モデルは呼ばない。
- `--iterations 1 --proposal-file outputs/proposal.json`: 現在のCodex sessionや既存autoresearchから、提案JSONだけを受け取るoffline接続。
- `--iterations N`: Codex CLIに1候補ずつ提案させる有限loop。既定3、1〜100。各提案は採用済みbestと直近6件のfeedbackを受け取る。
- 生成されるcodeは `# ORIGIN: ORIGINAL` で開始。全コードを返す。diffやMarkdown fenceを返さない。
- 実モデル呼出は `src/meta_search_rsi/llm.py` だけ。Codex exec read-only/ephemeral/構造化出力を利用する。
- 認証はoutputs/codex-homeのChatGPT login。globalのauth/configはコピーしない。provider API keyは要求しない。

`keep` / `discard` / `crash` を必ず記録。codeの構文/実行エラーはcrash、JSON契約違反はinvalid_proposal、CLI異常/timeoutはcodex_error。評価器の不変条件違反は候補の失敗として隠さず、直ちに停止する。

再開時は保存済みbestのtrial snapshotとSHA-256を確認する。過去prompt/trialは上書きしない。候補の取得途中で中断した場合、その候補は未評価のまま残し、次回は新番号から始める。bestは明示的な評価成功と改善を確認したときだけ更新する。

このcommandは開発用評価集合を使う。policyが各replay内で未観測結果を見ないことと、改善役のCodexが複数trialの評価feedbackを見ることは別である。未知taskへの一般化を主張する場合は別のholdout設計が必要。
