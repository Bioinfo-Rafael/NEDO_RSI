"""Single explicit entry point. No implicit whole-corpus inference."""
import argparse
from .db import audit, dumps


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    for name in ('audit','extract','select','delex','build','verify','excel-data','coverage'):
        a=s.add_parser(name)
        if name in ('select','delex'):a.add_argument('--limit',type=int,default=100 if name=='select' else 10)
        if name=='select':a.add_argument('--seed',type=int,default=17);a.add_argument('--max-chars',type=int,default=100000)
        if name in ('delex','build'):a.add_argument('--model',default='gpt-5.6-sol')
        if name=='delex':
            a.add_argument('--retries',type=int,default=2);a.add_argument('--timeout',type=float,default=300);a.add_argument('--batch-size',type=int,default=10)
            a.add_argument('--workers',type=int,default=1)
        if name=='verify':a.add_argument('--full-hash',action='store_true')
        if name=='coverage':a.add_argument('--source-db',required=True);a.add_argument('--node-id',required=True)
    args=p.parse_args()
    if args.command=='audit':result=[{k:v for k,v in r.items() if k in ('path','counts','runs_with_branch')} for r in audit()]
    elif args.command=='extract':
        from .compress import extract
        result=extract()
    elif args.command=='select':
        from .select import select
        result=select(args.limit,args.seed,args.max_chars);result={k:v for k,v in result.items() if k not in ('selected_ids','discarded')}
    elif args.command=='delex':
        from .delex import process
        result=process(args.limit,args.model,args.retries,args.timeout,args.batch_size,workers=args.workers);result={k:v for k,v in result.items() if k!='records'}
    else:
        from . import build
        if args.command=='build':result=build.build(args.model)
        elif args.command=='verify':result=build.verify(args.full_hash)
        elif args.command=='excel-data':result=build.excel_data()
        else:result=build.coverage(args.source_db,args.node_id)
    print(dumps(result))


if __name__=='__main__':main()
