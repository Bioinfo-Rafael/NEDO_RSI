"""Read-only raw inventory and integrity snapshot; never normalize source files."""
import hashlib
import json
from common import ROOT
from adapters.evo_mcts import EVAL,EvoMCTS
from adapters.search_agents import SearchAgents,pages
from adapters.treeofthoughts import TreeOfThoughts

def main():
    report=ROOT/'reports'; old=json.loads((report/'raw_integrity_complete.json').read_text())
    expected={x['path']:x for x in old}; rows=[]
    for p in sorted((ROOT/'raw').rglob('*')):
        if not p.is_file():continue
        key=str(p.relative_to(ROOT)); row={'path':key,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}
        if key in expected:assert row==expected[key],key
        rows.append(row)
    assert set(expected)<=set(x['path'] for x in rows)
    (report/'raw_integrity_complete.json').write_text(json.dumps(rows,indent=2))
    swe=json.loads((report/'swe_counts_all.json').read_text());rest=json.loads((report/'rest_counts.json').read_text())
    html=list(SearchAgents().discover());empty=0;page_count=0
    for p in html:
        _,records=pages(p);empty+=not records;page_count+=len(records)
    evo=list(EvoMCTS().discover())
    tot=list(TreeOfThoughts().discover())
    census={
        'openevolve':{'unit':'program checkpoint JSON snapshot','records':sum(1 for p in (ROOT/'raw/openevolve/repository').glob('**/programs/*.json')),'note':'同一UUIDのsnapshotをまとめる。metadata/issue attachmentsは別の補助raw。'},
        'searchagents':{'unit':'render HTML','records':len(html),'nonempty_files':len(html)-empty,'empty_files':empty,'rendered_pages':page_count,'note':'page内のcandidateも展開するためnode数とは一致しない。'},
        'evomcts':{'unit':'evaluation log record','records':sum(bool(EVAL.search(line)) for p in evo for line in p.read_text().splitlines()),'paper_records':38,'breakthrough_records':6,'note':'paper/breakthroughは既存evaluation nodeへの追加payload。'},
        'restmcts':{'unit':'published JSON/JSONL row','records':sum(x['records'] for x in rest),'files':len(rest),'note':'DPO rowからchosen/rejected各1 run。各pathを明示stepに分割。dataset間重複を含む。'},
        'sweagent':{'unit':'parquet trajectory row','records':swe['total_raw_rows'],'unique_trajectory_ids':swe['unique_trajectory_ids'],'demonstration_files':18,'note':'traj_idでencoding重複を除外。demonstrationは別run。'},
        'dreamrsi':{'unit':'published trajectory','records':0,'note':'公開実履歴なし。'},
        'treeofthoughts':{'unit':'search log top-level run','records':sum(len(json.loads(p.read_text())) for p in tot),'converted_files':len(tot),'note':'4 search logs対象。別の8 naive/cache JSONはraw保管のみ。'}
    }
    (report/'raw_record_census.json').write_text(json.dumps(census,ensure_ascii=False,indent=2))
    print(json.dumps({'raw_files':len(rows),'raw_bytes':sum(r['bytes'] for r in rows),'sources':census},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
