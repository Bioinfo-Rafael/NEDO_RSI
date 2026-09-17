# CodexでMeta Search RSIを改善する

このdirectory外の研究コード・DB・raw dataはread-only。原DB接続は既存db.pyのmode=roを使う。
`~/.codex`、global environment、shell初期化ファイルを変更しない。

## policy autoresearchを依頼されたとき

1. README.md、autoresearch/program.md、対象studyの採用policyとfeedbackを読む。
2. studyを指定し、回数を有限にする。最初は `./autoresearch.sh --study ... --dry-run`。
3. 通常のpolicy探索で変更可能な実装は `autoresearch/policy_candidate.py` のみ。評価器・config・tests・DB・memory・manifestを変更しない。
4. 現在のCodex sessionで候補を作る場合は、outputs/に `{"hypothesis":"...","code":"..."}` を保存し、`./autoresearch.sh --study ... --iterations 1 --proposal-file outputs/...json` で評価する。新しいCodexを起動する必要はない。
5. 自動運転は `./autoresearch.sh --study ... --iterations N`。Codexはコード提案だけを返し、hostがstudy内候補を検証する。原本candidateはseedとして保持する。
6. 改善しなければbestを変えない。crash・不正JSON・timeoutも記録する。同じstudyで再開できる。
7. raw scoreの異task平均、未観測結果の参照、run IDのhardcode、存在しないedge/結果の生成を禁止する。
8. 報告はobjective、採否、比較可能run数、保存先、development評価の限界を含める。

## 基盤そのものの変更

ユーザーがCLIや評価器等の改善を明示した場合はその変更を行える。その際はPAPER_MAP/PROVENANCE/READMEを更新し、testsを実行して新studyを作る。旧manifestを書き換えない。これはpolicy探索時の固定範囲とは別の保守作業。

## モデル呼び出し

追加のprovider SDKを導入しない。任意のCodex呼び出しはsrc/meta_search_rsi/llm.pyへ集約する。
実Codexの利用は認証・ネットワークと利用枠を使う。連携の開発テストはdry-runまたは偽CLIで行い、実モデルで実行したかどうかを区別する。
