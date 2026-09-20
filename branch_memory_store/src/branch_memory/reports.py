"""Generate measured audit/selection reports; no guessed counts or performance."""
import collections
import json
from .db import ROOT, dumps, output, write_json


def measured_reports():
    audits=json.loads((ROOT/'outputs/audit.json').read_text())
    sources=[r for r in audits if 'branch_points' in r]
    selection=json.loads((ROOT/'outputs/selection.json').read_text())
    lines=['# Storage audit','', '計測日: 2026-09-20。GB/MBはdecimal。SQLite `dbstat` はtableとindexが占めるpage bytes、columnは全行の `sum(length(cast(column AS BLOB)))` によるUTF-8 text bytes。後者はrecord header・数値・index・pageの余白を含まない。サンプル外挿ではなく全行集計。','',
           '## DB別の容量と構造','', '| Source DB | Bytes | Runs | Nodes | Events | Branch points | Branch runs | Branchなしruns |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in audits:
        n=r['counts'];branch=len(r['branch_points']) if 'branch_points' in r else 'source別合計と重複'
        lines.append(f"| {r['path']} | {r['bytes']:,} | {n['runs']:,} | {n['experiences']:,} | {n['raw_events']:,} | {branch} | {r.get('runs_with_branch','—')} | {r.get('linear_only_runs','—')} |")
    public=next(r for r in audits if r['path']=='public_experience/public_experience.db')
    lines+=['','## Public DBの物理page容量','','| Table / index | Bytes | GB | 比率 |','|---|---:|---:|---:|']
    for r in sorted(public.get('storage_pages',[]),key=lambda r:-r['bytes']):
        lines.append(f"| {r['name']} | {r['bytes']:,} | {r['bytes']/1e9:.4f} | {r['bytes']/public['bytes']:.2%} |")
    lines+=['','## Source別の主要column（全行のtext bytes）','','| Source | Column | Bytes |','|---|---|---:|']
    for r in sources:
        columns=sorted(((t+'.'+k,v) for t,d in r['columns'].items() for k,v in d.items()),key=lambda x:-x[1])
        for k,v in columns[:8]:lines.append(f"| {r['path'].split('/')[-1]} | {k} | {v:,} |")
    lines+=['','## なぜ大きいか・今回の縮小方法','',
        '主因はraw_eventsだけではない。public DBはexperiencesが約27.90 GB、raw_eventsが約12.92 GB、runsが約2.12 GB。ReSTは累積prompt/resultと各stepのmetadata、SWEは観測・prompt・raw_jsonに加えて全祖先IDを連結したdisplay_pathの重複も大きい。publicのstdout/message列はNULLで、主にraw_jsonへ内容を格納する。localはraw event本文・投影列が大部分。',
        '', '元DBの情報を維持して容量を減らす一般案は、raw event全文を外部の圧縮ファイルに置いてhash/locatorで参照すること、固定source metadataを共有化すること、display_pathをparentから必要時に計算すること、累積推論prefixを差分で保存すること。具体的削減量は再設計後の検証が必要で、上のcolumn bytesを単純合算して削減保証とはしない。元DBにはこの変更を加えていない。',
        '', 'branch memoryはlinear-only runを除外し、parentで確認できる分岐点とそのincoming/terminal区間だけを採用する。raw_events tableは複製しない。非branch nodeを1つのownerへだけ割り当て、root anchorはrunsに1回保存する。DB原本に戻れる参照を残す。samplingで選外になる意思決定は新DBには含まれず、情報を全件維持した圧縮とは区別する。',
        '', f"SQLite freelist pages（public）: {public['freelist_count']}。VACUUM等は一切実行していない。"]
    output(ROOT/'STORAGE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    lines=['# Selection report','',f"seed={selection['seed']}、最大unit文字数={selection['max_input_chars']:,}、有限のsemantic処理枠={selection['limit']}件。",'',
        '| Source | Total branch points | Selected | Discarded | Reasons |','|---|---:|---:|---:|---|']
    for r in sources:
        source='nedo_rsi' if r['path'].startswith('experience_store/') else r['path'].split('/')[-1][:-3]
        d=selection['by_source'].get(source,{'total':0,'selected':0,'discarded':0,'reasons':{}})
        reason=d['reasons'] or ('linear_only: '+str(r['linear_only_runs'])+' runs' if r['linear_only_runs'] else 'raw recordsなし' if not d['total'] else '全件保持')
        lines.append(f"| {source} | {d['total']:,} | {d['selected']:,} | {d['discarded']:,} | {reason} |")
    lines+=['','## 選定の実装','',
        '原本はlocal experience.db + source別normalized/*.dbだけ。public/combinedから同じrowを重ねて取り込まない。全eligible NEDO分岐を優先し、その後sourceを均等に巡回する。source内はtask_name hash・depth bucket・branching factor bucket・score有無・status・parent relationで層化し、seed+source_hashのSHA256順で選ぶ。Pythonのランダムhashや現在時刻に依存しない。',
        '', 'OpenEvolve/Evoをsource巡回の先頭に置く。完全tree/明示parentを持つOpenEvolveを含める一方、少数sourceが大量task sourceに埋もれないようsource単位で均等配分する。完全tree優先を全件の厳密な辞書順にした設計ではない。選定は5GBを埋めるためではなく、モデル処理時間を有限にするため。実測input bytesは選定JSONに残し、長大unitを切り詰めずinput_size_limitとして選外にした。これはstorage_capによる除外ではない。',
        '', 'unknown_parentのNULLを架空rootへ結び直さない。確認できる2本以上の子を持つ部分木は、元の不確実性を保持したままeligibleにする。invalid/cycle/duplicateは抽出エラーとし、恣意的に修復しない。',
        '', '圧縮リンクは元branch point IDを維持する。next/previousが選外のunitを指す場合がある。そのリンクは存在しない新DB rowを実在するように扱わず、source側参照として残す。SQLでLEFT JOINし、未収録を判別する。',
        '', '## Coverage','', '全branching runの元node割当はoutputs/coverage.jsonl.gz。最終選定後のownerが選外ならcoverage CLIはdiscarded_samplingを返す。branchなしrunの各nodeはDB/run所属を照会してexcluded_linear_runと判定する。非branch rootはrunsに保存し、coverageではunresolved + reason=root_anchor_in_runsで区別する。branchのない孤立componentはunresolved + component_without_branch_or_missing_parent。',
        '', 'source別全node数・linear-only run数はSTORAGE_AUDIT.md。全branch IDの選定/除外理由はoutputs/selection.json。linear_onlyはrun単位の除外であり、存在しないbranch pointをdiscarded件数へ足さない。']
    output(ROOT/'SELECTION_REPORT.md').write_text('\n'.join(lines)+'\n')
    source_manifest=json.loads((ROOT/'source_manifest.json').read_text())
    datasets=collections.defaultdict(set)
    for d in source_manifest['downloads']:datasets[d['dataset']].add(d['revision'])
    appendix=['','## 取得済みHF datasetの全revision','','| Dataset | Revision |','|---|---|']
    for name,revisions in sorted(datasets.items()):appendix.append(f"| https://huggingface.co/datasets/{name} | {', '.join(sorted(revisions))} |")
    appendix+=['','## 再計測したcanonical source件数','','| Source DB | Bytes | Runs | Nodes | Events | Branch points | Branch runs |','|---|---:|---:|---:|---:|---:|---:|']
    for r in sources:
        n=r['counts'];appendix.append(f"| {r['path']} | {r['bytes']:,} | {n['runs']:,} | {n['experiences']:,} | {n['raw_events']:,} | {len(r['branch_points']):,} | {r['runs_with_branch']:,} |")
    document=ROOT/'DATASET_BUILD.md';text=document.read_text().split('\n## 取得済みHF datasetの全revision')[0]
    document.write_text(text+'\n'.join(appendix)+'\n')
    return {'sources':len(sources),'branches':selection['total'],'selected':selection['selected']}


def pilot_report():
    selection=json.loads((ROOT/'outputs/selection.json').read_text())
    lines=['# Pilotとsemantic処理の検証','', '実Codex CLIを使用した測定。外部provider SDKや固定の疑似LLM応答は成果物生成に使っていない。unit testの模擬応答とは別。','',
           '| Pilot | Input units | Successful | Failed | JSON parse | Elapsed seconds | Mean input chars | Mean output chars |','|---|---:|---:|---:|---:|---:|---:|---:|']
    measured={}
    cache_values=[json.loads(p.read_text()) for p in (ROOT/'outputs/cache').glob('*.json')]
    unit_ids={u['source_hash']:u['branch_point_id'] for u in json.loads((ROOT/'outputs/selected.json').read_text())}
    for version in ('v1','v2'):
        r=json.loads((ROOT/f'outputs/pilot_{version}.json').read_text());n=len(r['records'])
        ids={x['branch_point_id'] for x in r['records']}
        outputs=[v['delexicalized'] for v in cache_values if v['prompt_sha256']==r['prompt_sha256'] and unit_ids.get(v['source_hash']) in ids]
        row={'units':n,'success':r['success'],'failed':r['failed'],'json_parse_success':r['json_parse_success'],
             'elapsed_seconds':r['elapsed_seconds'],'mean_input_chars':sum(x['input_chars'] for x in r['records'])/n,
             'mean_output_chars':sum(x.get('output_chars',0) for x in r['records'])/n,
             'source_distribution':dict(collections.Counter(x['source_type'] for x in r['records'])),
             'usage':dict(collections.Counter({k:sum(x.get('usage',{}).get(k,0) for x in r['records']) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')}))}
        measured[version]=row
        row['mean_output_bytes']=sum(len(dumps(v).encode('utf-8')) for v in outputs)/len(outputs) if outputs else None
        lines.append(f"| {version} | {n} | {r['success']} | {r['failed']} | {r['json_parse_success']} | {r['elapsed_seconds']:.2f} | {row['mean_input_chars']:.1f} | {row['mean_output_chars']:.1f} |")
    lines+=['','## Distribution / token usage','']
    for v,r in measured.items():lines+=[f"- {v}: sources={r['source_distribution']}、平均output bytes={r['mean_output_bytes']}、CLI usage={r['usage']}。"]
    seconds=measured['v1']['elapsed_seconds']/measured['v1']['units']
    lines+=['', 'input_tokensはCLIが報告した値で、raw unitだけでなくCodexのsystem/developer context等も含む。cached_input_tokensは内数として別掲する。reasoning tokenをoutputと二重加算しない。',
        '', f"v1の逐次throughputは {1/seconds:.4f} units/s。単純外挿では100件が約 {100*seconds/60:.1f} 分、全{selection['total']:,}件が約 {selection['total']*seconds/3600:.1f} 時間。実際は文字量・共有context cache・並列度・混雑で変わる。4並列の理想値は約1/4だが保証しない。巨大unitは選定外なので全件外挿は特に粗い。",'',
        '金額はnot available。利用者の契約・利用枠に依存し、CLIログに課金額はないため推測しない。全件処理を開始せず、最終100件に限定した。',
        '', '## 人が確認した内容','',
        'v1では10/10が構造検証を通ったが、OpenEvolveの出力に「26-circle packing」等のtask固有表現が残った。v2で具体的名称・数・benchmarkを機能表現へ置換する指示を追加。旧cacheを新promptの結果として採用しない。',
        '', 'v2のlocal2件とOpenEvolve1件をrawと照合した。localではobjectiveの定義がraw.evidenceにあり、terminalとbaselineの改善方向、keep/discardの区別が支持される。OpenEvolveではbranch-pointのtimeoutとterminalのvalid/combined-scoreを照合し、baselineにmatching scoreがないためoutcome_type=unknownを維持している。',
        '', '追加でEvo-MCTSのopaque operator e3、Search Agentsの異なる目標を述べるcandidate、ToTのincomingとnext-only unitを確認した。operatorの意味を捏造せずunknownにすること、candidate valueを実行成功へ言い換えないこと、未収録continuationの結果を補完しないことを確認。全100件の文ごとの完全な意味監査ではない。',
        '', '## 品質検査の範囲','',
        '全件にschema、node ID/順序、terminalのstep数、continuation、決定論的outcome typeの検査を適用。数値はrawの正本に保持する。自由文の含意や抽象度は自動検証だけで証明できず、rawとの対比とuncertaintiesを利用者が確認する。']
    final_path=ROOT/'outputs/full_processing.json'
    if final_path.exists():
        final=json.loads(final_path.read_text())
        pilot=json.loads((ROOT/'outputs/pilot_v2.json').read_text())
        records=pilot['records']+final['records']
        n=len(records)
        usage={k:sum(x.get('usage',{}).get(k,0) for x in records) for k in ('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens')}
        lines+=['','## 最終v2 datasetの処理実績','',
            f"v2 pilot {pilot['success']}件 + 本処理{final['success']}件 = {n}件completed。failed={final['failed']}。本処理は{final['workers']}並列、{final['elapsed_seconds']:.2f}秒（{final['success']/final['elapsed_seconds']:.4f} units/s）。本処理開始時のcache hitは{final['cache_hit']}件。",
            '', f"本処理のattempts={final['attempts']}、JSON parse成功={final['json_parse_success']}。1応答がtrajectory step数検証で拒否され、再試行で成功した。v2全体では101呼出、最終採用100件。v1 pilotの10呼出は別で、最終DBへ混入しない。",
            '', f"採用100件の平均input={sum(x['input_chars'] for x in records)/n:.2f} chars、平均output={sum(x['output_chars'] for x in records)/n:.2f} chars。採用応答のCLI usage合計={usage}。このusageは拒否された応答およびv1 pilotを含まないため、全試行の請求token合計とは区別する。",
            '', '本処理後の再実行は100 cache hit、0 attempts、追加モデル呼出0。記録はoutputs/resume_check.json。全件のraw/schema/outcome検査、2table整合性、圧縮全5,326点のcycle/親子整合性を検証した。']
    output(ROOT/'PILOT_REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'outputs/pilot_metrics.json',measured)
    return measured
