import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from . import analyze,verify
from .lengths import length_atlas
from .length_oracle import verify_lengths


def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d


def main():
    p=argparse.ArgumentParser(description='Bounded soft-DTW certificate generator and exhaustive verifier')
    p.add_argument('command',choices=['analyze','verify','lengths','verify-lengths']);p.add_argument('input',help='JSON file or - for stdin');p.add_argument('-o','--output')
    a=p.parse_args()
    try:
        with (sys.stdin.buffer if a.input=='-' else open(a.input,'rb')) as f:raw=f.read((4<<20)+1)
        if len(raw)>4<<20:raise ValueError('input exceeds 4 MiB')
        data=json.loads(raw,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
        if a.command=='analyze':value=analyze(data)
        elif a.command=='lengths':value=length_atlas(data)
        else:value={'verified':(verify(data)if a.command=='verify'else verify_lengths(data)),'schema':data['schema']}
        text=json.dumps(value,indent=2,allow_nan=False)+'\n'
        if a.output:
            dest=Path(a.output);name=None
            try:
                with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=dest.parent,delete=False)as f:name=f.name;f.write(text)
                os.replace(name,dest)
            finally:
                if name and os.path.exists(name):os.unlink(name)
        else:sys.stdout.write(text)
        return 0
    except (ValueError,TypeError,KeyError,OSError,RecursionError) as e:
        print('warplab: '+str(e),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
