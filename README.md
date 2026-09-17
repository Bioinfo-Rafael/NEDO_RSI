# autoresearchをローカルのCSVで試す

## リポジトリ構成

このrepositoryには、World Model研究コード、autoresearch実験コード、Graph of Trace連携、
Dream-RSI Experience Storeを収録しています。

| Directory | 内容 |
|---|---|
| `WorldModel_plants/` | 焼却炉World Model、制御評価、autoresearch実験コード |
| `autoresearch/` | autoresearch本体とローカルCSV実験コード |
| `Graph-of-Trace/` | Codexから直接trace nodeを書き込めるGoT MCP/Viewer |
| `dream_rsi_experience_store/` | Experience Storeのschema、変換、export、auditコード |
| `experience_store/` | 3-table SQLite DB、Excel、JSONL、監査結果 |

Experience Storeの探索木は、`experience_store/experience.xlsx`の`Experiences`シート、または
次のcommandで確認できます。

```bash
python3 dream_rsi_experience_store/scripts/print_trees.py
```

詳細は[Dream-RSI Experience Store README](dream_rsi_experience_store/README.md)を参照してください。

ローカルのCodex認証・session原本、仮想環境、cache、119MBの入力CSV、実験生成物は
repositoryへ含めていません。

Simulatorを使った固定LLMのWorld Model開発とA/B比較は、[WorldModel_plants/research/README.md](WorldModel_plants/research/README.md)を参照してください。CSV版と独立した実行入口で、データ生成・学習・予測と制御の評価・Agentログ解析まで進めます。

## 別のCodexチャットで改善実験を一通り実行する

同じローカルフォルダ`/Users/cls-lab/Git/NEDO_RSI`を作業先として新しいチャットを開き、下の指示を貼り付けて送信してください。以前の会話を読んでいることに依存せず、必要な情報をファイルから読み直す手順です。準備済みのCSV・仮想環境・未コミットの追加コードを使うため、この実行では同じローカルフォルダを使用します。同じ`train.py`を複数チャットから同時に編集しないでください。

```text
/Users/cls-lab/Git/NEDO_RSI を作業先として、README.mdと
autoresearch/csv_experiment/README.mdを読み、後者の
「別チャット用の開始指示」に従ってautoresearch方式の改善実験を一通り実行してください。
baselineを含め5回、コード変更・学習・検証・採否判断・最良コードの復元・最終レポートまで行ってください。
各実験の仮説、変更内容、検証RMSE/MAE、所要時間、採否をチャットで報告してください。
testは使わず、最後に比較表と実測値・予測値のグラフを保存してください。
計画の提示だけで止めず、通常の実験の間に確認を挟まず、指定回数まで進めてください。
```

今回の「一通り」は、準備済みの5分後の炉内温度予測で、AIが改善ループを最後まで回すことです。元のkarpathy版GPT学習の実行や、未実装の複数指標予測・policyの開発とは別です。

所要時間は5回でおおむね10〜30分を見込む仮の目安です。各学習・評価は最大300秒で、それにAIのコード変更・検討・レポート作成時間が加わります。元のRidgeの単発再実行は約1.6秒でしたが、変更後のモデルも同じ速さとは限りません。

## このリポジトリは何か

[karpathy/autoresearch](https://github.com/karpathy/autoresearch)は、AIエージェントが小型言語モデルの学習コードを変更し、学習・評価・変更の採否を繰り返す実験環境です。人は研究方針をMarkdownで記述し、エージェントが実験を進めます。単独で起動する自動研究サービスではなく、コードを読み書きしてコマンドを実行する外部のAIエージェントと組み合わせます。

主な構成は次の3ファイルです。

| ファイル | 役割 |
| --- | --- |
| `autoresearch/prepare.py` | テキストデータ取得、トークナイザ作成、データ読み込み、評価。実験中は固定 |
| `autoresearch/train.py` | GPT、最適化手法、学習処理。エージェントが改良する対象 |
| `autoresearch/program.md` | 実験の進め方、変更可能範囲、採否ルールを定める指示書 |

元の実験では学習部分に約5分の予算を与え、検証テキストの`val_bpb`（1バイトあたりの予測損失。小さいほどよい）を比較します。起動・コンパイル・評価時間は別にかかります。改善した変更を残し、悪化した変更を戻して次の仮説を試します。これは言語モデルの事前学習であり、設備制御向けの強化学習や表形式データのAutoMLが実装済みという意味ではありません。

## 今回の環境とCSV

- clone先：`autoresearch/`
- 確認したコミット：`228791fb499afffb54b46200aca536f79142f117`
- 環境：Apple M4、メモリ24 GB、macOS 26.5
- 入力：`200801-0831 (1).csv`（124,760,938 bytes、CP932）
- 内容：2020年8月1日〜31日、276,392行・121列。炉内温度、流量、O2濃度、操作値など。
- 一部のセンサ列に空欄あり。列の単位・計算方法・操作値の確定時刻は別途確認が必要です。

元コードはCUDAとFlash Attentionを使用し、依存関係もNVIDIA向けPyTorchを指定しています。このMacで元の`uv sync` / `uv run train.py`をそのまま実行することはできません。また、元の入力は`text`列を持つParquetであり、このCSVを直接読み込む機能はありません。

そのため、CSVを使う入口として`autoresearch/csv_experiment/`にCPUで動く時系列予測実験を追加しました。元レポの「実験コードを変更→固定データで評価→改善を採用」という考え方を応用したローカル拡張で、元のGPT学習の実行とは区別します。

初期の動作確認では、現在と過去のセンサ値から5分後の`炉内ガス温度平均`を予測しました。これは環境確認用の仮設定です。以下の研究目的全体を実装したものではありません。

## ユーザーが示した研究目的

1. 燃焼状態・熱回収・排ガスの時系列の将来予測。
2. 望ましい焼却状態を実現するために、供給・操作を選ぶpolicyの獲得。

### 予測モデル

まず「過去から現在までの観測・操作履歴 → 将来の複数指標」を学習します。対象候補は炉内温度・火床温度・燃切点位置、主蒸気流量、O2・CO・NOxなどです。NOxはCSV上で逆算値のため、実測値と区別して評価します。主蒸気流量は熱回収に関連する指標ですが、熱量そのものを算出するには蒸気条件などの追加情報が必要です。

policyの評価にも用いる段階では「履歴＋これから与える操作列 → 将来の状態列」という操作条件付きの予測モデルへ拡張します。実際の未来操作を入力する評価は、その操作が既知という条件の評価であり、未来操作が未知の通常の予測性能とは分けて報告します。過去ログからの予測精度だけでは、未実施の操作を行った効果を正しく予測できるとは限りません。

予測対象、予測時間、観測履歴長、操作更新間隔は未確定です。1・5・15分先など複数時間の比較を候補とし、指標ごとの誤差とピーク時の誤差を確認します。単位の異なる指標を生のRMSEのまま足し合わせず、学習区間の尺度や合意した許容誤差で正規化します。

### policy

目指すインターフェースは「観測・操作履歴＋目標と制約 → 次の供給・操作」です。目標例は炉温を所定範囲に保つ、蒸気流量を目標に近づける、CO・NOxを抑える、操作の急変を避けることです。数値目標や優先順位は未確定であり、ここでは設定しません。

初期の比較対象として、過去ログの操作を再現する模倣学習policyを検討します。ただし、ログへの一致度は望ましい焼却の達成度とは異なります。目標条件付きpolicy、予測モデルを使った操作候補の比較・最適化、offline RLを次の候補とします。予測モデルで操作列を最適化する方式はMPCとして構成でき、毎回の最適化なしで操作を出すpolicyが必要なら、その結果を別途学習する方法もあります。

CSVの流量・速度の測定値が、そのまま独立して設定できる操作量とは限りません。SV/MV変更列も実操作・推奨操作・注釈のどれか未確認です。操作対象、コード値の意味、変更値が絶対値か差分か、指示から測定までの時間差、許容範囲を確認してからactionを確定します。

既存運転ログの操作分布から離れたpolicyは、予測・評価の不確実性が増します。policyの模倣誤差、モデル内での目標達成、独立した環境での閉ループ評価は区別します。この段階はオフラインでの研究設計であり、実設備との接続は含みません。

### autoresearchとの対応

予測モデル用の実験では固定した分割・指標のもとでモデル構造や学習法を改善し、policy用の実験では目標・操作制約・評価環境を固定して改善します。二つの実験系列を分け、policyが評価に使う予測モデルを都合よく変更できないようにします。

現時点で実行済みなのは単一温度のRidge予測だけです。複数指標の予測、操作条件付きモデル、policyは未実装です。

参考：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)、[Learning to Reach Goals via Iterated Supervised Learning](https://arxiv.org/abs/1912.06088)。これらは一般的な手法の根拠であり、この焼却炉での有効性を示す資料ではありません。

## このMacでCSV版を実行する

Python 3.12の仮想環境・依存関係・CSVの前処理データを準備済みです。

```bash
cd /Users/cls-lab/Git/NEDO_RSI/autoresearch/csv_experiment
.venv/bin/python run.py --description "baselineの再実行"
```

1回の学習・検証を実行します。詳しいデータ処理、環境の再作成方法、結果の見方、AIへ改善を依頼する文面は[CSV版README](autoresearch/csv_experiment/README.md)に記載しています。

CSV版の役割は、`prepare.py`が固定の前処理、`train.py`が変更する回帰モデル、`run.py`が固定の実行・評価、`program.md`がエージェント用の実験指示です。元のGPT版の同名ファイルと実行ディレクトリを取り違えないでください。

## 元の言語モデル実験を実行する場合

対応するNVIDIA GPU環境にcloneし、元READMEに従って実行します。H100での検証が記載されていますが、必要メモリやカーネル互換性はGPU・設定ごとに確認してください。

```bash
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch
uv sync
uv run prepare.py
uv run train.py
```

この手順が使うのは標準のテキストデータで、手元のCSVではありません。Mac向けの言語モデル学習が目的なら、元READMEに掲載された[macOS版](https://github.com/miolini/autoresearch-macos)や[MLX版](https://github.com/trevin-creator/autoresearch-mlx)も別の選択肢です。これらのforkは今回clone・検証していません。

## 参照

- [元README](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/README.md)
- [元のデータ準備処理](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/prepare.py)
- [元の学習処理](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/train.py)
- [元の実験指示書](https://github.com/karpathy/autoresearch/blob/228791fb499afffb54b46200aca536f79142f117/program.md)
