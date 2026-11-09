"""Verify each inverse-response witness by independent complete-path sums."""
import math
from fractions import Fraction
from .inverse import inverse_request,SCHEMA,TOL,WIDTH
from .oracle import enumerate_paths,compare


def verify_inverse(report):
    if type(report)is not dict or 'request'not in report:raise ValueError('invalid inverse report')
    req=inverse_request(report['request']);r=req['alignment'];target=req['target'];n,m=len(r['x']),len(r['y']);g=r['gamma']
    paths=list(enumerate_paths(n,m));lengths=[len(p)for p in paths]
    costs=[float(sum((Fraction((r['x'][i]-r['y'][j])**2)for i,j in p),Fraction()))for p in paths]
    legal=list(range(max(n,m),n+m))
    def direct(penalty):
        energies=[c+penalty*k for c,k in zip(costs,lengths)];shift=min(energies);w=[math.exp(-(e-shift)/g)for e in energies];z=math.fsum(w);p=[v/z for v in w]
        mean=math.fsum(v*k for v,k in zip(p,lengths));var=math.fsum(v*(k-mean)**2 for v,k in zip(p,lengths))
        return dict(penalty=penalty,mean=mean,variance=var,derivative=-var/g,distribution=[dict(length=k,probability=math.fsum(v for v,l in zip(p,lengths)if l==k))for k in legal])
    endpoints=[direct(-4),direct(4)];lo,hi=endpoints
    if len(legal)==1:status='unidentifiable'if abs(lo['mean']-target)<=TOL else 'target_above_range'if target>lo['mean']else'target_below_range'
    elif target>lo['mean']+TOL:status='target_above_range'
    elif target<hi['mean']-TOL:status='target_below_range'
    elif lo['mean']-hi['mean']<=TOL:status='numerically_flat'
    elif abs(lo['mean']-target)<=TOL:status='lower_endpoint'
    elif abs(hi['mean']-target)<=TOL:status='upper_endpoint'
    else:status=None
    trace=report.get('trace');expected=[];a,b=-4.,4.
    if type(trace)is not list or len(trace)>40:raise ValueError('invalid trace size')
    if status is not None and trace:raise ValueError('trace forbidden for classified endpoint')
    if status is None:
        if not trace:raise ValueError('missing bisection trace')
        for i,step in enumerate(trace):
            mid=(a+b)/2;v=direct(mid);expected.append(dict(lower=a,upper=b,**v))
            done=abs(v['mean']-target)<=TOL or b-a<=WIDTH
            if done:
                if i!=len(trace)-1:raise ValueError('trace continues after stop')
                status='residual_tolerance'if abs(v['mean']-target)<=TOL else'bracket_width'
            if i==len(trace)-1 and not done:raise ValueError('trace stopped too early')
            if v['mean']>target:a=mid
            else:b=mid
    candidates=endpoints+[{k:v for k,v in s.items()if k not in ('lower','upper')}for s in expected]
    best=min(candidates,key=lambda v:(abs(v['mean']-target),v['penalty']));selected=None if status=='unidentifiable'else best
    value=dict(schema=SCHEMA,request=req,absolute_mean_tolerance=TOL,penalty_width_tolerance=WIDTH,status=status,legal_lengths=legal,endpoints=endpoints,trace=expected,selected=selected,residual=None if selected is None else selected['mean']-target)
    compare(value,report)
    return True
