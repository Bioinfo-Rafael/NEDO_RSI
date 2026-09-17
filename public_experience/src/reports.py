"""Create source mappings, human tree samples and an independent SQLite audit."""
import collections
import hashlib
import importlib.util
import io
import json
import sqlite3
import subprocess
from contextlib import redirect_stdout
from common import ROOT
from registry import ADAPTERS

DETAILS={
'openevolve':('OpenEvolve','openevolve.md','obtained','tree',
    'issue #156の本文・commentsから解決したypwang61/MyOpenEvolveをSHA固定clone。全program checkpointを取得。公式repoのreleasesには履歴assetなし。checkpoint関連issue 23件も調査し、#88の単一programと#141のlogを追加取得。',
    'Program JSONのid/parent_id/generation/iteration_found/code/metrics/prompts。異なるcheckpointで同一UUIDは同一nodeにまとめ、promptsを最も多く保持するsnapshotを採用し、全snapshotをraw_eventsに残す。親リンクの連結成分を1 runとする。',
    'prompt=parent code/metrics、proposal=changes_descriptionまたは保存済みLLM responses、result_summary=child code、score=metrics.combined_score。元ProgramDatabase.get_best_programのdescending順によりmaximize。',
    'repos/myopenevolve/openevolve/database.py: Program / _save_program / get_best_program; controller.py: child Program生成・log_prompts。',
    '5 root UUIDは別tree。snapshotを別nodeとして水増ししない。initial programsにはactionなし。combined_scoreのないnodeはNULL。#88/#141はrun/parentを確定できないためunbound raw_eventsとして保持。',
    'Apache-2.0（repository。個別issue添付の独立したlicense表記は未確認）'),
'searchagents':('Search Agents','search_agents.md','partial','partial_search_tree',
    'READMEの公開Google Drive gpt4o_search_trajectories.zip（2,189,522,114 B）を認証なしで取得。全910 HTMLとresults.txt。sitesはshopping/reddit/classifieds（VisualWebArena）。別WebArena archiveへの公開リンクは確認できなかった。',
    'RenderHelperのstate_obv/predict_action/additional_textをparse。候補にはa_idx,curr_a_idx,depth,scoreがある。選択済みbest_actionsは複数pageに同じcandidate一覧が繰り返されるため同じbatch内でdedup。',
    'prompt=実行pageの観測、proposal=raw_prediction、result_summary=選択action後の次page観測、score=sourceのvalue。初期観測は実観測nodeとして保持。task PASS/FAILはrun metadataとnode_id=NULLのraw_eventsへ保存し、候補rewardにしない。',
    'repos/search_agents/run.py: take_action_and_score、action_queue、all_candidates、best_actions; browser_env/helper_functions.py: RenderHelper.render。',
    'depth0の親はsearch開始時の実行node。深い候補は同a_idx・直前depthの親候補が一意のときのみ接続。curr_a_idxはlocal sibling indexでglobal parent IDではない。曖昧ならparent=NULL/parent_relation=unknown。未選択branchの観測は保存されていない。空HTMLはrawに残すがnodeを作らない。',
    'MIT（repository）。Drive archive内の独立したlicenseは未確認'),
'evomcts':('Evo-MCTS','evo_mcts.md','partial','partial_tree',
    '指定paper_data JSONL全38件、全5 production run logs、breakthrough JSON全6件。元物理dataset・モデル・評価器実行は取得/実行しない。',
    'paper JSONLはeval_times/depth/operator/thinking/reflection/code/fitness/algorithm。READMEはcomplete treeと呼ぶがparent/children/edgesは含まれない。logsの評価行とAction/Father Obj/Now Obj行を照合する。',
    'run=production log、node=eval_times、proposal=operator + 保存済みthinking/reflection、prompt=保存済みuser/system prompt（ある場合）、next state=保存済みcodeまたは評価結果。scoreはlogのobjectiveをそのまま保存（minimize）。paperの正のfitnessはmetrics_jsonに別保存し、符号反転しない。',
    'repos/evo_mcts/source/evo_mcts.py: expand/run/serialize_mcts_tree; evolution_interface.py: get_algorithm/evolve_algorithmのnp.round(objective,5); mcts.py: MCTSNode.parent。',
    'Father Objを公式と同じ5桁丸めで照合し、先行する同run・depth-1の候補が一意の場合だけparentを復元。restore_mcts_treeの「最後の同depth node」heuristicは使用しない。infはREAL scoreに入れずNULL、元のinfはrawに保存。未公開rootや欠落codeを創作しない。depthは保持edgeから計算し、元depthはmetadata.original_depth。',
    'GPL-3.0（repository）'),
'restmcts':('ReST-MCTS','rest_mcts.md','partial','linearized_mcts / linear_reasoning',
    'README Data & Modelの全20 Hugging Face datasets、全25 JSON files。PRM-0th/PRM-1st、3model×2round×3methodのpolicy/CoT/DPOを取得。モデルweightは取得しない。',
    'JSON配列、拡張子.jsonのJSONL、DPO column arrays。instruction/output、content/summary、prompt_answer/label、chosen/rejected、response_chosen/response_rejectedを扱う。',
    '1公開path=1 run。DPOはchosen/rejectedを別runとしchoiceをmetadataに保持。明示的Step n:の境界だけでnodeへ分割。prompt=question+prefix、proposal=step、result_summary=prefix+step。PRM labelは最終prefix nodeだけに保存し、中間stepへ配らない。無記録scoreはNULL。',
    'repos/rest_mcts/MCTS/base.py: treeNode.append_children/update_y_from_parent/get_full_value_samples*; MCTS/task.py: policy_samples/value_samples; utils/format_dpo.py; PRM/train_VM_chatglm.py。',
    '公開training pathsから元MCTSの完全treeや共通ancestor IDは復元できない。同じquestion/prefixでも異なる探索runを勝手にmergeしない。PRMはsourceが付けたprocess labelであり、今回生成したrewardではない。CoT/DPO baselineはlinear_reasoningとしてMCTSと区別。空solutionにもlabelがある場合、action/resultはNULL。重複する公開dataset/fileは別出典として保持しており、独立な実験数ではない。',
    'PRM-0th: Apache-2.0。他19 datasets: CC-BY-4.0（各HF cardData）。code repositoryのlicenseは未確認'),
'sweagent':('SWE-agent / SWE-smith','swe_agent.md','obtained','linear',
    'SWE-agent v1.1.0 release→SWE-smith→公式HF trajectoriesを追跡。tool/xml/ticks/trainの全32 parquet、repoの18 demonstration .trajを取得。repository snapshot/Docker imageは取得しない。',
    '.trajのtrajectory[thought,action,observation]、HF messages/instance_id/resolved/model/traj_id/patch。messagesはtool/xml/ticksでJSON文字列、trainで配列。traj_idが同じencoding違いはtool→xml→ticks→trainの順で1つ採用し、すべての元encodingはrawに残す。',
    '1 trajectory=1 run、1 assistant action turn=1 node。parentは直前turn。prompt=直前observation（初回はissue/system context）、proposal=tool_callsまたはassistant content、result_summary=次のassistantまでのobservations。prefix全体はmetadata内のfile,row,message indexで参照する。元messagesにnode/parent IDはない。node_id末尾とmetadata.original_parent_idはassistant turn番号、message_index/original_node_idは元messages配列の位置。親の元位置はparent_idで親nodeのmessage_indexを読む。結合には共通node_id/parent_idを使う。',
    'repos/swe_agent/docs/usage/trajectories.md、sweagent/agent/agents.pyのtrajectory記録。HFのpinned README/featuresと全parquet schemaを確認。',
    'resolvedはbooleanのまま最終node.metrics_json/run metadataに保存し、新しい0/1 rewardにしない。公開sourceに数値scoreがないためscore=NULL。最後のsubmit後のobservationが未保存ならNULL。返されたsource patchを検証・修正・実行していない。',
    'MIT（repository/Hugging Face dataset）'),
'dreamrsi':('Dream-RSI','dream_rsi.md','unavailable','unavailable',
    'zhengkid/Dream-RSIのcurrent tree・README・releases・project pageを再確認。SHA 4149ea9181ab1db80f85717ffda2c9f0f130e85b。',
    'repositoryにはpaper/assets/README/release planのみ。実discovery history/replay worlds/checkpoints/trajectory dataは見つからない。',
    'adapterは0 recordsを返す。demoや論文の図を実行履歴に見立てない。',
    'reports/inventory/dream_rsi/{README.md,tree.json,releases.json}; https://www.dream-rsi.com/',
    'codeもrelease準備中。公開実履歴の代わりにsynthetic dataを作成しない。',
    '未確認'),
'treeofthoughts':('Tree of Thoughts（追加）','treeofthoughts.md','obtained','beam_search_forest / dfs_search_forest',
    '追加探索2候補中、princeton-nlp/tree-of-thought-llmを採用。全12 log/cache JSONを取得。変換対象は2 beam logsと2 DFS logsの4 files（240 search runs）。他8 filesはnaive baseline/cacheとしてrawにのみ保持。',
    'beam: steps[].ys/new_ys/values/select_new_ys。DFS: actions[],env_step,total_step,info.r_word/r_letter。',
    'beamはofficial生成処理がparent prefix+new textを返すことを確認し、直前levelの選択prefixが一意のときだけ親にする。DFSはactions[:-1]と一致するstackが一意なら親にする。score=valueまたはr_wordをそのまま保存、maximize。',
    'repos/treeofthoughts/src/tot/methods/bfs.py: get_samples/get_proposals/solve; scripts/crosswords/search_crosswords-dfs.ipynb: dfsのactions.append/popとinfo保存。',
    'prompt-only rootはlogにないため人工nodeを追加しない。rootの子群はforestとして保持。重複prefixで親が一意に決まらない場合はNULL。LATSも調査したが、saved implementations/feedbackは全parentを保存しないうえaccは累積task精度なのでnode scoreとして採用しなかった。',
    'MIT（repository）'),
}

def pct(n,d):return f'{100*n/d:.2f}' if d else '—'

def main():
    reports=ROOT/'reports';stats={}
    census=json.loads((reports/'raw_record_census.json').read_text())
    for source in ADAPTERS:
        m=json.loads((reports/f'{source}_audit.json').read_text());stats[source]=m
        title,filename,status,structure,obtained,fmt,mapping,code,limits,license_=DETAILS[source]
        n=m.get('nodes',0);edges=m.get('edges',0);h=m.get('branching_histogram',{});branching=sum(v for k,v in h.items() if int(k)>1)
        lines=[f'# {title}','','Status: **'+status+'**','', '## 取得内容','',obtained,'','## Raw format / tree','',fmt,'','## 共通schemaへのmapping','',mapping,'','## 実装根拠','',code,'','## 変換結果','',
            '| 指標 | 値 |','|---|---:|',f"| Runs | {m.get('runs',0):,} |",f'| Nodes | {n:,} |',f'| Edges | {edges:,} |',f"| parent=NULL | {m.get('roots',0):,} |",f"| Max stored depth | {m.get('max_depth',0)} |",f'| 2 children以上のnode | {branching:,} |',f"| 最大children数 | {max(map(int,h),default=0)} |",f"| Raw events | {m.get('raw_events',0):,} |",f"| Parent coverage (edges/nodes) | {pct(edges,n)}% |",f"| Action coverage | {pct(m.get('action',0),n)}% |",f"| Result coverage | {pct(m.get('result',0),n)}% |",f"| Score coverage | {pct(m.get('score',0),n)}% |",f"| Unknown parent relations | {m.get('unknown_parents',0):,} |",'','## 復元不能項目・制約','',limits,'','## License','',license_,'','## Development / held-out','',
            'run_idのSHA256 modulo 5で分割。全件監査のsource別JSONに件数・coverage・hard failuresを保存。問題内容が重複するreleaseもあるため独立な問題汎化性能の評価ではない。','',f'詳細: `{source}_audit.json`。']
        raw=census[source]
        lines += ['', '## Raw records found', '', f"{raw['records']:,} records（単位: {raw['unit']}）。{raw['note']}",
                  '', '## Integrity / provenance', '', f"Raw provenance coverage: {pct(m.get('provenance',0),n)}%。",
                  f"cycles={m.get('cycles',0)}, duplicate IDs={m.get('duplicate_ids',0)}, invalid parents={m.get('invalid_parents',0)}, unsupported scores={m.get('fake_scores',0)}。統合後のorphan/FK検査はfinal_audit.md。",'',
                  'children数ごとのnode件数: `'+json.dumps(h,sort_keys=True)+'`。']
        (reports/filename).write_text('\n'.join(lines)+'\n')
    original=ROOT.parent/'experience_store/experience.db'
    old=sqlite3.connect(f'file:{original}?mode=ro',uri=True)
    expected=old.execute("SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()
    checks={};counts={}
    for name in ('public_experience','combined_experience'):
        print('auditing',name,flush=True)
        c=sqlite3.connect(f'file:{ROOT/(name+".db")}?mode=ro',uri=True)
        c.execute('PRAGMA cache_size=-2097152')
        assert c.execute("SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name").fetchall()==expected
        counts[name]={t:c.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('runs','experiences','raw_events')}
        for table,pk in (('runs','run_id'),('experiences','node_id'),('raw_events','event_id')):
            assert c.execute(f"SELECT count(*) FROM {table} WHERE {pk} IS NULL OR {pk}='' ").fetchone()[0]==0
        sample=json.loads((reports/'excel_sample_manifest.json').read_text())
        assert counts[name]==sample['public_total' if name=='public_experience' else 'combined_total']
        if name=='public_experience':
            selected=sample['selected_public_run_ids'];params=','.join('?' for _ in selected)
            for table in ('runs','experiences','raw_events'):
                assert c.execute(f'SELECT count(*) FROM {table} WHERE run_id IN ({params})',selected).fetchone()[0]==sample['public_included'][table]
        # Strictly increasing depth on every edge proves acyclicity without recursive SQL.
        invalid=c.execute('SELECT count(*) FROM experiences a LEFT JOIN experiences p ON a.parent_id=p.node_id WHERE a.parent_id IS NOT NULL AND (p.node_id IS NULL OR a.run_id!=p.run_id OR a.depth!=p.depth+1)').fetchone()[0]
        orphan,orphan_runs,mismatched_events=c.execute('''SELECT
            COALESCE(SUM(e.node_id IS NOT NULL AND n.node_id IS NULL),0),
            COALESCE(SUM(e.run_id IS NOT NULL AND r.run_id IS NULL),0),
            COALESCE(SUM(e.run_id IS NOT NULL AND n.node_id IS NOT NULL AND e.run_id!=n.run_id),0)
            FROM raw_events e LEFT JOIN experiences n ON e.node_id=n.node_id
            LEFT JOIN runs r ON e.run_id=r.run_id''').fetchone()
        invalid_roots=c.execute('SELECT count(*) FROM runs r LEFT JOIN experiences n ON r.root_node_id=n.node_id WHERE r.root_node_id IS NOT NULL AND (n.node_id IS NULL OR n.run_id!=r.run_id OR n.parent_id IS NOT NULL)').fetchone()[0]
        assert invalid==orphan==orphan_runs==mismatched_events==invalid_roots==0
        fk=c.execute('PRAGMA foreign_key_check').fetchall();assert not fk
        checks[name]={'same_schema':True,'invalid_parent_or_depth':invalid,'orphan_events':orphan,'orphan_event_runs':orphan_runs,'mismatched_event_runs':mismatched_events,'invalid_roots':invalid_roots,'foreign_key_errors':len(fk),'cycles':0,'duplicate_node_id':0}
        if name=='combined_experience':
            keys={'runs':'run_id','experiences':'node_id','raw_events':'event_id'}
            for t,pk in keys.items():
                cols=[r[1] for r in old.execute(f'PRAGMA table_info({t})')];ki=cols.index(pk)
                for row in old.execute(f'SELECT * FROM {t}'):
                    assert c.execute(f'SELECT * FROM {t} WHERE {pk}=?',(row[ki],)).fetchone()==row,(t,row[ki])
            checks[name]['all_original_rows_identical']=True
        c.close()
        print('verified',name,counts[name],checks[name],flush=True)
    old.close()
    protected=json.loads((reports/'protected_hashes.json').read_text())
    for p,h in protected.items():assert hashlib.file_digest((ROOT.parent/p).open('rb'),'sha256').hexdigest()==h
    manifest=json.loads((reports/'raw_integrity_complete.json').read_text());changed=[]
    by_path={x['path']:x for x in manifest}
    for filename in ('download_manifest.json','download_manifest_swe_additional.json'):
        for downloaded in json.loads((reports/filename).read_text()):
            assert not downloaded.get('error')
            assert by_path[downloaded['path']]['sha256']==downloaded['sha256']
    for x in manifest:
        p=ROOT/x['path']
        if p.stat().st_size!=x['bytes'] or hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=x['sha256']:changed.append(x['path'])
    assert not changed
    totals={k:sum(m.get(k,0) for m in stats.values()) for k in ('runs','nodes','edges','roots','action','result','score','raw_events','unknown_parents')}
    assert counts['public_experience']=={'runs':totals['runs'],'experiences':totals['nodes'],'raw_events':totals['raw_events']}
    audit={'sources':stats,'totals':totals,'db_counts':counts,'checks':checks,'raw_files':len(manifest),'raw_bytes':sum(x['bytes'] for x in manifest),'source_raw_modified':0,'protected_files_unchanged':True}
    (reports/'final_audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# Final audit','','実SQLiteへのread-only検査結果。元DB・Excelを再生成していない。','', '| Source | Runs | Nodes | Edges | Root* | Max depth | Parent % | Action % | Result % | Score % | Structure |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for s,m in stats.items():
        n=m.get('nodes',0)
        lines.append(f"| {DETAILS[s][0]} | {m.get('runs',0):,} | {n:,} | {m.get('edges',0):,} | {m.get('roots',0):,} | {m.get('max_depth',0)} | {pct(m.get('edges',0),n)} | {pct(m.get('action',0),n)} | {pct(m.get('result',0),n)} | {pct(m.get('score',0),n)} | {DETAILS[s][3]} |")
    n=totals['nodes'];lines.append(f"| TOTAL | {totals['runs']:,} | {n:,} | {totals['edges']:,} | {totals['roots']:,} | {max(m.get('max_depth',0) for m in stats.values())} | {pct(totals['edges'],n)} | {pct(totals['action'],n)} | {pct(totals['result'],n)} | {pct(totals['score'],n)} | mixed |")
    lines += ['', '*Rootは`parent_id IS NULL`の件数。真のrootに加えて公開情報から親を復元できないnodeを含む。',f"親関係unknownは **{totals['unknown_parents']:,}** 件。invalid parent=0は、完全な木を復元できたことを意味しない。",'', '## 検証','', '- cycles = 0（すべてのedgeでchild.depth=parent.depth+1、run一致）','- duplicate node_id = 0（PRIMARY KEYと全source統合INSERTで確認）','- invalid parents = 0 / orphan events = 0 / foreign key errors = 0','- schema = 元experience.dbと全table/index SQL一致','- source raw modified = 0（全rawのSHA-256照合）','- 元DB・Excel = SHA-256一致、combined内の元全row = 全column一致','- 数値score = nodeごとにraw event上の出典値と一致を全件build時評価。sourceのないscoreはNULL','',f"Raw保管: {len(manifest):,} files / {audit['raw_bytes']:,} bytes（ZIPとその展開物を両方含む）。",'','## Excel','', 'ユーザー選択により3シートのrun単位サンプル。全件数・収録数・省略数・run ID・長文セルの扱いは[excel_sample.md](excel_sample.md)。DBは全件。','', '## 不完全なデータ','', 'ReST-MCTSは公開training pathで、full search treeではない。SWE-agentはlinear。Evo-MCTSとSearch Agentsには未復元parentや未公開観測がある。詳細はsource別reportとunavailable_sources.md。']
    (reports/'final_audit.md').write_text('\n'.join(lines)+'\n')
    download=json.loads((reports/'download_totals.json').read_text())
    with (reports/'final_audit.md').open('a') as f:
        f.write(f"\n## 取得容量の定義\n\n履歴rawの取得対象は {download['canonical_raw_files']:,} files / {download['canonical_raw_bytes']:,} bytes。ZIP展開前の値で、cloneから取り出した履歴ファイルを含みます。展開物を含む保管数は上記Raw保管の値です。repoコード・.git・生成DBの容量は含みません。詳細: download_totals.json。\n")
    write_inventory(stats)
    write_tree_samples()
    write_readme(stats,totals)
    print(json.dumps({'totals':totals,'checks':checks,'raw_files':len(manifest),'raw_bytes':audit['raw_bytes']},ensure_ascii=False),flush=True)

def write_inventory(stats):
    lines=['# Source inventory','','調査日: 2026-09-17。主要6 source + 追加2候補を調査、追加採用はTree of Thoughtsのみ。','', '| Source | Status | Repo/data | Structure | Nodes | License |','|---|---|---|---|---:|---|']
    urls={'openevolve':'https://github.com/algorithmicsuperintelligence/openevolve/issues/156','searchagents':'https://github.com/kohjingyu/search-agents#agent-trajectories','evomcts':'https://github.com/iphysresearch/evo-mcts','restmcts':'https://github.com/THUDM/ReST-MCTS#data--model','sweagent':'https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories','dreamrsi':'https://github.com/zhengkid/Dream-RSI','treeofthoughts':'https://github.com/princeton-nlp/tree-of-thought-llm/tree/master/logs'}
    for s,d in DETAILS.items():lines.append(f'| {d[0]} | {d[2]} | {urls[s]} | {d[3]} | {stats[s].get("nodes",0):,} | {d[-1]} |')
    lines += ['', '追加候補LATS: https://github.com/lapisrocks/LanguageAgentTreeSearch をcloneし実装・saved logsを調査。全tree edgeが保存されておらず、累積accをnode rewardと混同しないため今回は未採用。上限5 datasetsに対し採用1件。', '', '全repoのcommit SHA、release、file count、HF dataset revision・sizeはinventory.json / additional_inventory.json / hf_inventory.json。', '', 'HF各datasetのlicenseはhf_inventory.json.info.cardData.licenseに保存。rawの再配布権を一律にMIT等へ変更していない。']
    (ROOT/'reports/source_inventory.md').write_text('\n'.join(lines)+'\n')

def write_tree_samples():
    path=ROOT/'normalized';lines=[]
    for source in ADAPTERS:
        c=sqlite3.connect(f'file:{path/(source+".db")}?mode=ro',uri=True);c.row_factory=sqlite3.Row
        run=c.execute('SELECT * FROM runs ORDER BY run_id LIMIT 1').fetchone()
        lines.append('\n'+DETAILS[source][0])
        if not run:lines.append('unavailable: 公開実履歴なし');c.close();continue
        lines.append('run_id='+run['run_id'])
        rows=list(c.execute('SELECT node_id,parent_id,proposal,score FROM experiences WHERE run_id=? ORDER BY sequence_index',(run['run_id'],)))
        children=collections.defaultdict(list)
        for n in rows:children[n['parent_id']].append(n)
        shown=0
        def visit(n,prefix='',last=True,root=False):
            nonlocal shown
            if shown>=80:return
            shown+=1
            label=' '.join((n['proposal'] or '[action NULL]').split())[:90]
            lines.append(prefix+('' if root else '└── ' if last else '├── ')+n['node_id'].removeprefix(run['run_id']+'::')+f" score={n['score']} "+label)
            kids=children[n['node_id']]
            for i,child in enumerate(kids):visit(child,prefix+('' if root else '    ' if last else '│   '),i==len(kids)-1)
        for n in children[None]:visit(n,root=True)
        if len(rows)>shown:lines.append(f'... {len(rows)-shown} nodes omitted from display; full tree is in DB')
        lines.append('注意: parent=NULLは真のrootと未復元parentの両方を含み得る。')
        c.close()
    (ROOT/'reports/tree_samples.txt').write_text('\n'.join(lines)+'\n')

def write_readme(stats,totals):
    lines=['# 公開探索履歴 Experience Store','','公開された探索履歴を、既存Experience Storeと同じ3テーブルへ変換したものです。モデル学習やLLM API呼び出しは行いません。','', '```text','Public search histories → source adapters → Runs / Experiences / Raw_Events → public_experience.db','```','', '- `Run` = 1本の探索履歴 / trajectory','- `Experience` = 1 node、`parent_id` = 復元できた親へのedge','- `prompt` = state/context、`proposal` = action、`result_summary` = next state/outcome','- `score` = sourceに記録された数値のみ。欠落はNULL','', '## 読むファイル','', '- `public_experience.db`: 公開データ全件','- `combined_experience.db`: 元の自前データを全columnそのままcopy + 公開データ','- `public_experience.xlsx` / `combined_experience.xlsx`: `Runs / Experiences / Raw_Events`の3シート。run単位サンプル。全件・収録・省略件数は[Excel範囲](reports/excel_sample.md)','- [木のサンプル](reports/tree_samples.txt) / [最終監査](reports/final_audit.md) / [取得元・license](reports/source_inventory.md)','', '## 変換結果','', '| Source | Status | Runs | Nodes |','|---|---|---:|---:|']
    for s,m in stats.items():lines.append(f"| [{DETAILS[s][0]}](reports/{DETAILS[s][1]}) | {DETAILS[s][2]} | {m.get('runs',0):,} | {m.get('nodes',0):,} |")
    lines += ['', '取得済みの履歴にも欠落があります。`parent_relation=unknown`は親不明であり真のrootとは限りません。`depth`は保存したedgeからの深さ、元depthはmetadataに保持します。複数rootのrunでは`root_node_id=NULL`、root一覧はrun metadataです。ReSTはpath、SWEはlinearで、完全な分岐探索木と同一視しないでください。','', '## 探索木を見る','', 'NEDO_RSI直下で実行します。まずrun IDを選びます。','', '```bash', "sqlite3 public_experience/public_experience.db 'SELECT run_id,source_type FROM runs LIMIT 10;'", 'python3 dream_rsi_experience_store/scripts/print_trees.py \\', '  --db public_experience/public_experience.db \\', '  --run-id openevolve::MyOpenEvolve::21d972d0-0b7d-49dd-8f88-f98a26dbc6ee','```','', '全runの無指定表示は出力が巨大になるため、`--run-id`を付けてください。GoT Viewer用got.jsonへの変換は今回行っていません。','', '## Pythonから読む','', '```python','import json, sqlite3','db = sqlite3.connect("file:public_experience/public_experience.db?mode=ro", uri=True)','db.row_factory = sqlite3.Row','run_id = db.execute("SELECT run_id FROM runs WHERE run_id GLOB ? ORDER BY run_id LIMIT 1", ("openevolve::*",)).fetchone()[0]','for node in db.execute("SELECT * FROM experiences WHERE run_id=? ORDER BY sequence_index", (run_id,)):','    meta = json.loads(node["metadata_json"])','    print(node["node_id"], node["parent_id"], node["score"], meta["parent_relation"])','db.close()','```','', 'SQLiteはPython標準sqlite3だけで読めます。公開行のsource_file/source_files_jsonの相対pathはpublic_experience/基準です。combinedにコピーした既存local行のpathは元仕様のまま保持しています。元データへの場所はsource_reference/source_files_json、より細かいJSON/parquet row/HTML page位置はmetadata_jsonにあります。raw_events.source_lineはJSONLでは行、JSON配列/parquet/HTMLでは1始まりのrecord/page番号です。','', '## 再現とparser改善','', '```bash','cd public_experience','python3 -m venv .venv','.venv/bin/python -m pip install -r requirements.txt','.venv/bin/python -m unittest discover -s tests -v','.venv/bin/python src/autoresearch.py --label next-candidate --source openevolve','.venv/bin/python src/build_db.py --source openevolve','```','', '取得計画は[download_plan.md](reports/download_plan.md)。inventory.py / external_inventory.pyは取得前metadata、download.py / download_search_agents.py / prepare_raw.pyはraw取得。全sourceのDB生成後は `src/build_db.py --combine` → `src/prepare_excel.py` → `src/export_excel.mjs` → `src/validate_excel.py` → `src/reports.py` の順です。Python scriptsは`.venv/bin/python`で実行します。Excel出力はCodex runtimeのartifact-toolを使い、`node src/export_excel.mjs public_experience` と `node src/export_excel.mjs combined_experience` を実行します。モデルやsource内の実験コードは実行しません。','', 'parserの変更許可範囲・評価方法は[program.md](program.md)、試行記録は[autoresearch_report.md](reports/autoresearch_report.md)。', '', '既存 `experience_store/experience.db` / `.xlsx` はread-onlyで、SHA-256が一致しています。今回のraw/repos/DBはlocal Git exclude対象で、この作業ではpushしていません。']
    (ROOT/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
