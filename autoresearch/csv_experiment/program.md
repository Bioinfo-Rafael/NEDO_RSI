# CSV時系列予測の実験指示

これはローカル追加実験専用です。親ディレクトリのprogram.mdはGPT/CUDA版なので、この実験には使いません。

## 目的と固定条件

現在と過去の観測から5分後の炉内ガス温度平均を予測し、validationのRMSEを下げる。

開始時は`../../README.md`で全体の研究目的を確認し、このディレクトリのREADME.md、program.md、config.json、prepare.py、run.py、train.py、requirements.txt、setup.sh、test_prepare.py、data/profile.jsonを読む。既存results/results.jsonlがあれば過去の指標も確認する。過去のチャット履歴を前提にしない。元CSV・NPZ・仮想環境を全件テキストとして読み込む必要はない。

- 実験コードで変更できるファイルは、このディレクトリの`train.py`のみ。例外としてresults/内への実験記録・最終レポート・グラフ・描画用スクリプトの作成を許可する。既存の実験結果は上書きしない。
- `fit_model(X_train, y_train)`が学習済み推定器を返すインターフェースを維持する。
- 学習に使用できるのは渡されたX_trainとy_trainだけ。検証・test・元CSVをtrain.pyから読まない。
- prepare.py、run.py、config.json、requirements.txt、data/、test_prepare.pyは実験中固定。
- 1回の実験は`.venv/bin/python run.py --description "仮説の説明"`。標準上限300秒を変更しない。
- 目的変数やラグ・分割などを変える場合は別の研究課題として扱い、過去のスコアと同列比較しない。
- `--split test`はモデル選択が終わった後に人が明示したときだけ使う。testの値を探索判断に使わない。
- パッケージ追加、ネットワークアクセス、CSVのアップロード、実設備への接続は実験の範囲に含まれない。

## 有限回の改善ループ

1. 既存のユーザー変更を確認し、現在のtrain.pyを初期状態として保存する。環境・準備済みデータの存在を確認して、まずそのまま1回実行し今回のbaselineを得る。既存データを理由なく再生成しない。baseline自体に失敗した場合は固定条件を勝手に変更せず、原因と必要な対応を報告する。
2. 仮説を1つ立ててtrain.pyを変更し実行する。例：Ridgeの正則化、HistGradientBoostingRegressorへの変更。
3. results/<run_id>/result.jsonのstatus、validation RMSE、所要時間を確認する。
4. 改善したら採用する。悪化・失敗・timeoutなら、自分が変更したtrain.pyだけを直前の最良状態に戻す。リポジトリ全体へのgit reset/cleanは使わない。
5. 各実験のrun_id、仮説、採否、理由をresults/decisions.mdへ追記する（実験記録のみ変更可能）。自動保存されるresults.jsonlや過去の実験は書き換えない。
6. ユーザーが指定した回数で止める。指定がなければbaselineを含め5回まで。失敗・timeoutも1回として数える。各回の開始時に回数・仮説・変更点、終了時にRMSE・MAE・今回baselineとの差・所要時間・採否をチャットへ報告する。実験が長引く場合は進行状況を報告する。
7. 最良のtrain.pyを残し、results/reports/<今回固有のID>/report.mdへ全実験の比較表、最良run_id、採用理由、単純な現在値据え置き予測との差、再実行コマンド、残る問題を保存する。モデルが改善しなければbaselineを残す。今回の実験と過去セッションの実験を区別する。
8. 最良run_idの保存済みmodel.joblibを読み、validation.npzのみで実測値・モデル予測・現在値据え置き予測を同じ時間軸に描く。全期間の概要と検証期間冒頭24時間の図を保存する。描画コードは同じレポートディレクトリに残す。依存追加は行わず、利用可能な描画環境がなければPython標準ライブラリでSVGを生成する。表示のため間引く場合は明記し、スコア自体は全validationで算出された値を報告する。
9. 最終レポートとグラフへのリンクをチャットで返す。最良コードの復元・報告用の予測・描画は新しい学習実験に数えない。通常の実験間に確認を挟まず、指定された有限回と最終報告まで完了する。

この指示書だけで常駐プロセスが起動するわけではありません。AIエージェントへの実験依頼が別途必要です。
