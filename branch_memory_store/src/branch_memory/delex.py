"""Bounded Codex JSON calls, strict grounding checks, cache and retry checkpoints."""
import collections
import concurrent.futures
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import threading
from .db import ROOT, dumps, sha, write_json, output

PROMPT_VERSION='branch-semantic-v2'
DEFAULT_MODEL='gpt-5.6-sol'
ACTIVE=set()
ACTIVE_LOCK=threading.Lock()
STOP=threading.Event()


def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}


TEXT={'type':['string','null']}
STRING={'type':'string'}
STRINGS={'type':'array','items':STRING}
SCHEMA=obj({'abstract_state':TEXT,'incoming_history':{'type':'array','items':obj({'node_id':STRING,'abstract_action':TEXT,'abstract_outcome':TEXT})},
            'decision_context':TEXT,'terminal_branches':{'type':'array','items':obj({'node_ids':STRINGS,'abstract_strategy':TEXT,
                'abstract_trajectory':STRINGS,'abstract_outcome':TEXT,'outcome_type':{'type':'string','enum':['improved','degraded','failed','neutral','unknown']}})},
            'continuations':{'type':'array','items':obj({'next_branch_point_id':STRING,'relationship':{'type':'string','enum':['continues_to_next_decision_point']}})},
            'search_pattern':TEXT,'uncertainties':STRINGS})


def outcome(base, leaf):
    if leaf.get('status') in ('failed','evaluation_failed'):
        return 'failed'
    if (base.get('score') is not None and leaf.get('score') is not None and base.get('score_name')
        and base.get('score_name')==leaf.get('score_name') and base.get('score_direction')==leaf.get('score_direction')
        and base['score_direction'] in ('minimize','maximize')):
        d=leaf['score']-base['score']
        if not d:return 'neutral'
        return 'improved' if (d>0)==(base['score_direction']=='maximize') else 'degraded'
    return 'unknown'


def validate_schema(value, schema=SCHEMA):
    types=schema['type'];types=[types] if isinstance(types,str) else types
    actual='null' if value is None else 'object' if isinstance(value,dict) else 'array' if isinstance(value,list) else 'string' if isinstance(value,str) else 'invalid'
    if actual not in types:raise ValueError('Invalid JSON type: '+actual)
    if 'enum' in schema and value not in schema['enum']:raise ValueError('Invalid enum')
    if actual=='object':
        if set(value)!=set(schema['required']):raise ValueError('Missing/extra JSON fields')
        for k,v in value.items():validate_schema(v,schema['properties'][k])
    elif actual=='array':
        for v in value:validate_schema(v,schema['items'])


def validate(value, unit):
    validate_schema(value)
    if [x['node_id'] for x in value['incoming_history']]!=unit['incoming_node_ids']:
        raise ValueError('Incoming evidence IDs/order changed')
    if [x['node_ids'] for x in value['terminal_branches']]!=unit['terminal_branch_node_ids']:
        raise ValueError('Terminal evidence IDs/order changed')
    for b,raw in zip(value['terminal_branches'],unit['terminal_branches_raw']):
        if len(b['abstract_trajectory'])!=len(raw):raise ValueError('Missing or invented trajectory steps')
        if b['outcome_type']!=outcome(unit['raw_context']['branch_point'],raw[-1]):raise ValueError('Unsupported outcome type')
    if [c['next_branch_point_id'] for c in value['continuations']]!=unit['next_branch_point_ids']:
        raise ValueError('Invented/missing continuations')
    text=dumps(value).lower()
    if re.search(r'\b(always superior|universally|caused|therefore proved)\b',text):
        raise ValueError('Unsupported causal/universal language')
    return value


def model_input(unit, run):
    anchor=unit['raw_context']['root_anchor_id']
    return {'unit':unit,'root_anchor':run['root_anchors'].get(anchor),
            'expected_terminal_outcome_types':[outcome(unit['raw_context']['branch_point'],path[-1]) for path in unit['terminal_branches_raw']]}


def cache_key(data, prompt, model):
    return sha(dumps({'input_sha256':sha(dumps(data)),'prompt_sha256':sha(prompt),'model':model,
                     'schema_sha256':sha(dumps(SCHEMA)),'reasoning_effort':'medium'}))


def cached(unit, run, model=DEFAULT_MODEL):
    prompt=(ROOT/'prompts/delexicalize.md').read_text()
    key=cache_key(model_input(unit,run),prompt,model)
    p=ROOT/'outputs/cache'/f'{key}.json'
    if not p.exists():return None
    result=json.loads(p.read_text())
    if result['cache_key']!=key or result['output_sha256']!=sha(dumps(result['delexicalized'])):
        raise ValueError('Corrupt cache')
    validate(result['delexicalized'],unit)
    return result


def invoke(data, directory, prompt, model, timeout, executable='codex'):
    home=ROOT/'outputs/codex-home'
    if not (home/'auth.json').exists():
        raise RuntimeError('Dedicated Codex home is not authenticated; see REPRODUCE.md')
    write_json(directory/'schema.json',SCHEMA)
    command=[executable,'exec','--sandbox','read-only','--ephemeral','--ignore-user-config','--skip-git-repo-check',
             '--color','never','--json','--model',model,'-c','model_reasoning_effort="medium"',
             '-c','cli_auth_credentials_store="file"','-c','forced_login_method="chatgpt"',
             '--cd',str(directory),'--output-schema',str(directory/'schema.json'),'-o',str(directory/'result.json'),'-']
    environment=dict(os.environ,CODEX_HOME=str(home),PYTHONDONTWRITEBYTECODE='1')
    for key in ('OPENAI_API_KEY','CODEX_API_KEY','ANTHROPIC_API_KEY'):environment.pop(key,None)
    write_json(directory/'request.json',{'command':command,'input_sha256':sha(dumps(data)),'model':model})
    with (directory/'events.jsonl').open('w') as events,(directory/'stderr.log').open('w') as errors:
        with ACTIVE_LOCK:
            if STOP.is_set():raise RuntimeError('Processing interrupted')
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=events,stderr=errors,text=True,env=environment,start_new_session=True)
            ACTIVE.add(process)
        try:process.communicate(prompt+'\n\nEVIDENCE JSON\n'+dumps(data),timeout=timeout)
        except BaseException:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait();raise
        finally:
            with ACTIVE_LOCK:ACTIVE.discard(process)
    if process.returncode:raise RuntimeError(f'Codex exit {process.returncode}; inspect {directory}/stderr.log')
    usage={}
    for line in (directory/'events.jsonl').read_text().splitlines():
        event=json.loads(line)
        if event.get('type')=='turn.completed':usage=event.get('usage',{})
    return json.loads((directory/'result.json').read_text()),usage


def work_unit(index, unit, run, prompt, version, model, retries, timeout, batch_size, executable):
    data=model_input(unit,run);key=cache_key(data,prompt,model)
    record={'branch_point_id':unit['branch_point_id'],'source_type':unit['source_type'],'input_chars':len(dumps(data)),
            'status':'failed','attempts':0,'json_parse_success':0}
    for retry in range(retries+1):
        if STOP.is_set():break
        batch=ROOT/'outputs/delex'/f'batch_{index//batch_size+1:06d}'/key
        attempt=1
        while (batch/f'attempt_{attempt:03d}').exists():attempt+=1
        directory=output(batch/f'attempt_{attempt:03d}'/'input.json').parent
        write_json(directory/'input.json',data)
        record['attempts']+=1;t=time.monotonic()
        try:
            value,usage=invoke(data,directory,prompt,model,timeout,executable)
            record['json_parse_success']+=1
            validate(value,unit)
            result={'delexicalized':value,'model':model,'codex_version':version,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'input_sha256':sha(dumps(data)),'source_hash':unit['source_hash'],'prompt_sha256':sha(prompt),
                    'delex_prompt_version':PROMPT_VERSION,'cache_key':key,'output_sha256':sha(dumps(value)),
                    'schema_sha256':sha(dumps(SCHEMA)),'reasoning_effort':'medium','usage':usage,'elapsed_seconds':time.monotonic()-t}
            write_json(ROOT/'outputs/cache'/f'{key}.json',result)
            record.update(status='completed',output_chars=len(dumps(value)),usage=usage,elapsed_seconds=result['elapsed_seconds'])
            break
        except (ValueError,RuntimeError,OSError,subprocess.TimeoutExpired) as e:
            record['error']=str(e)
            write_json(directory/'failure.json',{'error':str(e),'retry':retry,'elapsed_seconds':time.monotonic()-t})
            if isinstance(e,RuntimeError):break
    return record


def process(limit=10, model=DEFAULT_MODEL, retries=2, timeout=300, batch_size=10, executable='codex', workers=1):
    if not 1<=limit<=10000 or not 1<=batch_size<=100 or not 0<=retries<=5 or not 1<=workers<=4:
        raise ValueError('Invalid processing bounds')
    units=json.loads((ROOT/'outputs/selected.json').read_text())
    runs={r['run']['run_id']:r for r in json.loads((ROOT/'outputs/runs.json').read_text())}
    prompt=(ROOT/'prompts/delexicalize.md').read_text()
    version=subprocess.check_output([executable,'--version'],text=True).strip()
    report={'requested_limit':limit,'model':model,'prompt_version':PROMPT_VERSION,'prompt_sha256':sha(prompt),'codex_version':version,
            'success':0,'failed':0,'cache_hit':0,'attempts':0,'json_parse_success':0,'records':[],'workers':workers}
    start=time.monotonic();jobs=[]
    with output(ROOT/'outputs/delex.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        STOP.clear()
        for index,unit in enumerate(units):
            run=runs[unit['run_id']]
            if cached(unit,run,model):report['cache_hit']+=1;continue
            if len(jobs)<limit:jobs.append((index,unit,run))
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures=[pool.submit(work_unit,i,u,r,prompt,version,model,retries,timeout,batch_size,executable) for i,u,r in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                if future.cancelled():continue
                record=future.result()
                report['success' if record['status']=='completed' else 'failed']+=1
                report['attempts']+=record['attempts'];report['json_parse_success']+=record['json_parse_success']
                report['records'].append(record);report['elapsed_seconds']=time.monotonic()-start
                write_json(ROOT/'outputs/delex_latest.json',report)
                with output(ROOT/'outputs/delex_history.jsonl').open('a') as history:history.write(dumps(record)+'\n')
                print(dumps({k:record.get(k) for k in ('source_type','status','input_chars','output_chars','elapsed_seconds','error')}),flush=True)
                if record['status']=='failed' and 'Codex exit' in record.get('error',''):
                    STOP.set()
                    for f in futures:f.cancel()
        except BaseException:
            STOP.set()
            with ACTIVE_LOCK:
                for proc in ACTIVE:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
    report['elapsed_seconds']=time.monotonic()-start
    write_json(ROOT/'outputs/delex_latest.json',report)
    return report
