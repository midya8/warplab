"""Bounded inverse expected-length response with a numerical bisection witness."""
import math
from .core import request,number
from .lengths import length_atlas

SCHEMA='warplab-inverse-v1'
TOL=1e-9
WIDTH=1e-9


def inverse_request(data):
    if type(data)is not dict or set(data)!={'alignment','target'}:
        raise ValueError('expected exactly alignment and target')
    return dict(alignment=request(data['alignment']),target=number(data['target'],1,13))


def response(records,g,penalty):
    logs=[s['log_partition']-penalty*s['length']/g for s in records];top=max(logs)
    w=[math.exp(v-top)for v in logs];z=math.fsum(w);p=[v/z for v in w]
    mean=math.fsum(v*s['length']for v,s in zip(p,records))
    var=math.fsum(v*(s['length']-mean)**2 for v,s in zip(p,records))
    return dict(penalty=penalty,mean=mean,variance=var,derivative=-var/g,distribution=[dict(length=s['length'],probability=v)for s,v in zip(records,p)])


def solve_inverse(data):
    req=inverse_request(data);r=req['alignment'];target=req['target']
    atlas=length_atlas(dict(alignment=r));records=atlas['lengths'];g=r['gamma']
    low,high=response(records,g,-4),response(records,g,4);trace=[]
    if len(records)==1:
        status='unidentifiable'if abs(low['mean']-target)<=TOL else 'target_above_range'if target>low['mean']else'target_below_range'
    elif target>low['mean']+TOL:status='target_above_range'
    elif target<high['mean']-TOL:status='target_below_range'
    elif low['mean']-high['mean']<=TOL:status='numerically_flat'
    elif abs(low['mean']-target)<=TOL:status='lower_endpoint'
    elif abs(high['mean']-target)<=TOL:status='upper_endpoint'
    else:
        status='bracket_width'
        a,b=-4.,4.
        for _ in range(40):
            mid=(a+b)/2;v=response(records,g,mid);trace.append(dict(lower=a,upper=b,**v))
            if abs(v['mean']-target)<=TOL:status='residual_tolerance';break
            if b-a<=WIDTH:break
            if v['mean']>target:a=mid
            else:b=mid
    candidates=[low,high]+[{k:v for k,v in step.items()if k not in ('lower','upper')}for step in trace]
    best=min(candidates,key=lambda v:(abs(v['mean']-target),v['penalty']))
    # A single legal length has no identifiable penalty. Do not imply a unique estimate.
    selected=None if status=='unidentifiable'else best
    return dict(schema=SCHEMA,request=req,absolute_mean_tolerance=TOL,penalty_width_tolerance=WIDTH,status=status,legal_lengths=[s['length']for s in records],endpoints=[low,high],trace=trace,selected=selected,residual=None if selected is None else selected['mean']-target)
