# 分岐単位のsemantic delexicalization

目的は、保存された探索木の意思決定点について「そこに至る履歴」「観測された探索方向」「その後の実結果」を取り出し、別taskでも検索できる表現へ変換すること。未知actionの生成、原因推定、将来結果の予測はしない。

## 分岐点と区間の割当

同run内に実在する子が2つ以上あるnodeをbranch pointとする。parent_idのNULLは真のrootとも親不明とも限らない。元のparent関係だけを使い、branchがないrunを除く。元の孤立componentを人工rootで結ばない。

```text
root
  |
  A
  |
  B
 / \
C   D
|   |
E   F
|  / \
G H   I
```

| Branch point | incoming（端点を除く） | terminal paths | previous | next |
|---|---|---|---|---|
| B | [A] | [[C,E,G]] | NULL | [F] |
| F | [D] | [[H],[I]] | B | [] |

圧縮relationはB→F。rootはruns.root_anchors_jsonへ1回保存し、branch rowにはしない。root自体が2つ以上の子を持つ場合は通常のbranch rowとなる。

incomingは直前branch point（または非branch root）と現在branch pointの両端を除く。branch point自身はraw_context_json.branch_pointへ保存。次branchまでの一本道は**到達先**のincomingへ割り当てる。次branchに至らずleafで終わる一本道だけを出発点のterminalへ割り当てる。terminalにはleafを含む。根拠のない親やcycleを補修して記録しない。

`compress.py:compress` が検出と割当を行い、全nodeの分類を返す。B/Fのexact fixture、root分岐、linear-only、partial forest、cycle、重複、raw ID保持はtestsで検査する。元source内にあるancestor code等のfield内重複までは改変しないが、同一node recordを複数ownerへ複製しない。

## rawと抽象結果

1 branching point = branch_pointsの1row。rawは以下へ保持する。

- `raw_context_json`: branch pointの元ID、親、prompt、proposal、result_summary、score/name/direction、metrics、status、parent確度、score evidence。
- `incoming_raw_json` / `terminal_branches_raw_json`: 各segment内の同じfield群。元ID列は別JSON列にも保存。
- `source_reference`: 元DBの相対path、run ID、node ID、元source locator。
- `runs.root_anchors_json`: 非branch rootのraw情報。shared anchorを複製しない。

raw_events本文全体・画像・ZIPは保存しない。元DBに戻ってeventやartifact詳細を読む。`source_hash`は圧縮unit（rawと関係）の正規JSON hash。`input_sha256`はCodexへ渡したunit＋anchor＋決定論的outcome種別のhash。

## Codexへの入力と出力

モデル呼出境界は `delex.py:invoke`。Pythonで1unitを切り出した後にだけCodexへ渡す。`prompts/delexicalize.md` を版管理し、出力は `delex.py:SCHEMA` のJSON Schemaへ固定する。schema自体のhashもcacheに含める。各attempt directoryには実際のinput、schema、command、JSONLイベント、stdout最終結果、stderrがある。

```json
{
  "abstract_state": "Observed task-independent state, or null",
  "incoming_history": [
    {"node_id":"original ID", "abstract_action":"...", "abstract_outcome":null}
  ],
  "decision_context": "Observed choice context, or null",
  "terminal_branches": [
    {
      "node_ids":["original step ID"],
      "abstract_strategy":"Functional search direction",
      "abstract_trajectory":["One abstraction for each actual step"],
      "abstract_outcome":null,
      "outcome_type":"unknown"
    }
  ],
  "continuations":[
    {"next_branch_point_id":"original branch ID", "relationship":"continues_to_next_decision_point"}
  ],
  "search_pattern":"Structure observed at this decision point",
  "uncertainties":["Missing evidence or uncertain lineage"]
}
```

上は形の例であり、存在しないstepは埋めない。incoming/terminal/continuationがないときは空配列。text fieldはNULL可。余分なfieldを拒否する。incomingのID/順序、terminalの全ID/順序とstep数、next ID/順序は元unitと完全一致させる。

`outcome_type` はimproved/degraded/failed/neutral/unknown。元statusがfailed/evaluation_failedならfailed。それ以外はbranch pointとterminal leafのscoreが両方存在し、同metric名・同direction（minimize/maximize）の場合だけ比較する。不明directionや欠落scoreはunknown。原点はbranch pointであり、途中incumbentの採否とは違う。discarded candidateでもbaselineに比べればimprovedのことがある。数値scoreはrawにだけ保存し、モデルに新しいscoreを出させない。

semantic textの正しさをJSON Schemaだけで保証はできない。モデルが出力した文を人が原データと照合する。全recordの完全な意味監査を済ませたとは扱わない。

## model・prompt・実行設定

今回のmodelは`gpt-5.6-sol`、CLIは`codex-cli 0.154.0`。reasoning effortはmedium。temperatureは指定していない（このwrapperのCLI optionとして提供していない）。model名、Codex版、UTC timestamp、input/prompt/output/schema hashを結果cacheへ保存し、最終DBへ主要fieldを転記する。

最初の10件をv1でpilotし、task固有の語句が残ることを確認してv2へ改訂。旧promptはoutputs/prompt_v1.md、旧結果は異なるhashのcacheとして残す。最終DBには現prompt/version/modelに一致するcacheだけを採用する。

batch size=10はcheckpoint directoryの単位であり、10件を1つのpromptへ混ぜることではない。1モデル呼出には1unitだけ渡す。workers既定1、指定可能範囲1〜4。workersを増やしても選定集合とunit割当は変わらないが、model応答自体の完全な決定性は保証しない。

専用CODEX_HOMEはoutputs/codex-home。既存認証を利用する場合も秘密を含むauthはこのGit除外directory内だけに置く。実行はread-only sandbox / ephemeral / ignore-user-configを指定する。promptはtoolsやfile読み取りを禁止し、実行ログのtool使用も検査する。入力raw内の命令を実行しない。

CLIの構造化出力は[OpenAI公式non-interactive仕様](https://developers.openai.com/codex/noninteractive)と手元の`codex exec --help`で確認した。API keyや別provider SDKを追加しない。Codex実行には認証・通信・利用枠が必要。

## retry / checkpoint / resume

cache keyはinput、prompt、schema、model、reasoning設定のhash。completedかつhashとschemaが検証できたunitは再実行しない。別prompt/modelの結果を流用しない。`--limit`は新たに処理する未完了unit数で、cache hitはその枠を使わない。

JSON parse失敗、schema違反、node IDやoutcomeの不一致、timeoutは最大2回retry（合計3attempt）。非zero CLI終了は認証・通信・モデル利用可否の問題の可能性があり、そのcallを失敗として記録し未開始jobを停止する。部分結果をcompletedへ昇格させない。

`outputs/delex/batch_NNNNNN/<cache-key>/attempt_NNN/` に各attemptを保持。cacheは一時ファイルからatomic rename。同時runnerはfile lockで拒否する。Ctrl-Cとtimeout時には起動したprocess groupを停止する。再開時はcompleted cacheから続け、未完成attemptを上書きしない。

## sampling・限界

固定seed17の100件。全分岐5,326件の約1.88%であり、全分岐memoryではない。最大100,000文字のunit上限で大きなunitを選外とする。原文を無断で途中切断してcompletedにはしない。source別詳細はSELECTION_REPORT.md。

partial tree、未保存の候補観測、localの推定親子関係を含む。Search AgentsのvalueやToTのvalueは評価器の記録であり、現実taskの成功確率ではない。search_patternはそのunitで観測された構造だけを述べ、普遍則や因果として使わない。モデルの抽象化にも誤り得るため、利用時にrawと出典を確認する。
