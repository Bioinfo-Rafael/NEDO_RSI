"""Run-complete, stratified Excel sample. Full rows remain in SQLite."""
import collections
import argparse
import json
import re
import sqlite3
from common import ROOT,dumps

SHEETS={'runs':'Runs','experiences':'Experiences','raw_events':'Raw_Events'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--normalized',action='store_true',help='Read completed source tables before final merge; only in-memory views are created')
    args=parser.parse_args()
    source_totals=None
    if args.normalized:
        from registry import ADAPTERS
        public=sqlite3.connect(':memory:');aliases=[]
        source_totals={table:0 for table in SHEETS}
        for i,source in enumerate(ADAPTERS):
            path=ROOT/'normalized'/f'{source}.building.db'
            if not path.exists():path=ROOT/'normalized'/f'{source}.db'
            alias=f'source{i}';aliases.append(alias)
            public.execute(f'ATTACH DATABASE ? AS {alias}',(f'file:{path}?mode=ro',))
            indices=public.execute(f"SELECT count(*) FROM {alias}.sqlite_master WHERE type='index' AND sql IS NOT NULL").fetchone()[0]
            if indices!=5:raise RuntimeError(f'Source has not finished data/index creation: {source}')
            for table in SHEETS:
                source_totals[table]+=public.execute(f'SELECT count(*) FROM {alias}.{table}').fetchone()[0]
        for table in SHEETS:
            public.execute(f'CREATE TEMP VIEW {table} AS '+ ' UNION ALL '.join(f'SELECT * FROM {alias}.{table}' for alias in aliases))
    else:public=sqlite3.connect(f'file:{ROOT/"public_experience.db"}?mode=ro',uri=True)
    public.row_factory=sqlite3.Row
    selected=[];groups=collections.Counter();samples={};totals=source_totals or {t:public.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in SHEETS}
    print('counts',totals,flush=True)
    for row in public.execute('SELECT * FROM runs'):
        m=json.loads(row['metadata_json']);source=row['source_type'];dataset=m['source_dataset']
        group=(source,dataset,m.get('original_split',''))
        if source=='searchagents':group=(source,dataset,row['run_id'].split('::')[1])
        limit=10 if source in ('openevolve','evomcts') else 3
        if groups[group]>=limit:continue
        groups[group]+=1;selected.append(row['run_id']);samples[row['run_id']]=dataset
    print('selected complete runs',len(selected),flush=True)
    matrices={t:[] for t in SHEETS};columns={t:[r[1] for r in public.execute(f'PRAGMA table_info({t})')] for t in SHEETS}
    for rid in selected:
        for t in SHEETS:
            order='sequence_index,node_id' if t=='experiences' else 'sequence_no,event_id' if t=='raw_events' else 'run_id'
            matrices[t].extend(dict(r) for r in public.execute(f'SELECT * FROM {t} WHERE run_id=? ORDER BY {order}',(rid,)))
    counts={t:len(v) for t,v in matrices.items()};public.close()
    summary={'selection':'source/dataset/original_splitごとに先頭3 complete runs。OpenEvolve/Evo-MCTSは全run。Search Agentsはsiteごとに3 runs。',
             'public_total':totals,'public_included':counts,'public_omitted':{t:totals[t]-counts[t] for t in SHEETS},'selected_public_run_ids':selected,
             'long_cell_policy':'32767 UTF-16 code unitsを超えるtextはDB参照へ置換。値の省略はxlsxだけ。DBとrawは全文保存。','replaced_cells':{}}
    def write(name,data):
        sheets={};replaced=[]
        for t,title in SHEETS.items():
            original=columns[t];front={'runs':['source_type','source_dataset','run_id','project_name','root_node_id','status'],
                                     'experiences':['source_type','source_dataset','run_id','node_id','parent_id','depth','sequence_index','score','score_name','proposal','prompt','result_summary'],
                                     'raw_events':['source_dataset','run_id','node_id','event_id','event_type','sequence_no']}[t]
            cols=front+[c for c in original if c not in front];rows=[]
            for d in data[t]:
                ds=json.loads(d.get('metadata_json') or '{}').get('source_dataset') or samples.get(d.get('run_id')) or 'local'
                d={**d,'source_dataset':ds};pk=d.get({'runs':'run_id','experiences':'node_id','raw_events':'event_id'}[t]);values=[]
                for c in cols:
                    v=d.get(c)
                    if isinstance(v,str):
                        v=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]','',v)
                        if len(v.encode('utf-16-le'))//2>32767:
                            replaced.append({'table':t,'id':pk,'column':c})
                            v=f'[全文は {name}.db / {t} / ID={pk} / column={c} を参照]\n'+v[:1000]
                        if v.startswith('='):v="'"+v
                        if re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?',v):v="'"+v
                    values.append(v)
                rows.append(values)
            sheets[title]={'columns':cols,'rows':rows}
        output=ROOT/'.excel_data';output.mkdir(exist_ok=True)
        (output/(name+'.json')).write_text(dumps({'sheets':sheets}))
        summary['replaced_cells'][name]=replaced
    write('public_experience',matrices)
    local=sqlite3.connect(f'file:{ROOT.parent/"experience_store/experience.db"}?mode=ro',uri=True);local.row_factory=sqlite3.Row
    combined={t:[dict(r) for r in local.execute(f'SELECT * FROM {t}')]+matrices[t] for t in SHEETS}
    write('combined_experience',combined)
    local_counts={t:local.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in SHEETS};local.close()
    summary['combined_total']={t:totals[t]+local_counts[t] for t in SHEETS}
    summary['combined_included']={t:len(combined[t]) for t in SHEETS}
    summary['combined_omitted']=summary['public_omitted']
    (ROOT/'reports/excel_sample_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    md=['# Excelサンプルの範囲','',summary['selection'],'','既存localデータはcombined Excelに全件収録。public DBは全件、Excelはrun単位サンプル。','',
        '| Table | Public全件 | Excel収録 | 省略 | Combined全件 | Excel収録 |','|---|---:|---:|---:|---:|---:|']
    for t in SHEETS:md.append(f"| {t} | {totals[t]:,} | {counts[t]:,} | {totals[t]-counts[t]:,} | {summary['combined_total'][t]:,} | {len(combined[t]):,} |")
    md += ['',summary['long_cell_policy'],'',f"長文セル置換数: public={len(summary['replaced_cells']['public_experience'])}, combined={len(summary['replaced_cells']['combined_experience'])}。対象IDはexcel_sample_manifest.json。",'', 'source_datasetはmetadata_jsonから表示用に展開。DB schemaは変更していない。', '', "ISO日時にはExcelの自動変換・小数秒丸めを防ぐため、文字列保護用の先頭 `'` を付けています。DBの日時は元の文字列のままです。XLSXから読む場合はこの先頭1文字だけを除いてください。XMLで使えない制御文字はExcel表示だけで除去します。"]
    md += ['', 'NULLと空文字は、Excelではともに空白セルとして表示します。区別が必要な処理にはDBを使用してください。']
    (ROOT/'reports/excel_sample.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selected_public_run_ids','replaced_cells')},ensure_ascii=False))

if __name__=='__main__':main()
