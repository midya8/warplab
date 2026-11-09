"""Independent complete-path grouping and direct penalty-weighted sums."""
import math
from fractions import Fraction
from .lengths import length_request,SCHEMA
from .oracle import enumerate_paths,oracle,compare,verify


def _summarize(paths,costs,exact,weights,r):
    n,m=len(r['x']),len(r['y']);g=r['gamma'];z=math.fsum(weights);p=[v/z for v in weights]
    sets=[set(map(tuple,path))for path in paths];es=[set(zip(map(tuple,path),map(tuple,path[1:])))for path in paths]
    occ=[[math.fsum(v for v,s in zip(p,sets)if(i,j)in s)for j in range(m)]for i in range(n)]
    flows=[]
    for i in range(n):
        for j in range(m):
            for a,b in ((0,1),(1,0),(1,1)):
                if i+a<n and j+b<m:
                    pair=((i,j),(i+a,j+b));flows.append(dict(source=list(pair[0]),target=list(pair[1]),probability=math.fsum(v for v,s in zip(p,es)if pair in s)))
    cost=math.fsum(v*a for v,a in zip(p,costs));entropy=-math.fsum(v*math.log(v)for v in p if v);minimum=min(exact);winners=[path for path,e in zip(paths,exact)if e==minimum]
    grads=dict(gradient_x=[2*math.fsum(occ[i][j]*(a-b)for j,b in enumerate(r['y']))for i,a in enumerate(r['x'])],gradient_y=[2*math.fsum(occ[i][j]*(b-a)for i,a in enumerate(r['x']))for j,b in enumerate(r['y'])])
    return dict(expected_cost=cost,entropy=entropy,temperature_gradient=-entropy,occupancy=occ,edges=flows,hard=dict(score=float(minimum),ties=len(winners),path=min(winners)),**grads),p


def length_oracle(data):
    req=length_request(data);r=req['alignment'];g=r['gamma'];penalty=req['penalty'];n,m=len(r['x']),len(r['y']);c=[[(x-y)**2 for y in r['y']]for x in r['x']]
    paths=list(enumerate_paths(n,m));exact=[sum((Fraction(c[i][j])for i,j in path),Fraction(0))for path in paths];costs=list(map(float,exact));records=[]
    for length in range(max(n,m),n+m):
        ids=[k for k,p in enumerate(paths)if len(p)==length];part=[paths[k]for k in ids];cc=[costs[k]for k in ids];ee=[exact[k]for k in ids];shift=min(cc);weights=[math.exp(-(v-shift)/g)for v in cc]
        values,_=_summarize(part,cc,ee,weights,r);score=shift-g*math.log(math.fsum(weights));records.append(dict(length=length,count=len(part),log_partition=-score/g,score=score,**values))
    shifted=[cost+penalty*len(path)for cost,path in zip(costs,paths)];minimum=min(shifted);weights=[math.exp(-(v-minimum)/g)for v in shifted]
    values,probabilities=_summarize(paths,costs,exact,weights,r);mean=math.fsum(p*len(path)for p,path in zip(probabilities,paths));variance=math.fsum(p*(len(path)-mean)**2 for p,path in zip(probabilities,paths));distribution=[dict(length=rec['length'],probability=math.fsum(p for p,path in zip(probabilities,paths)if len(path)==rec['length']))for rec in records];score=minimum-g*math.log(math.fsum(weights));entropy_length=-math.fsum(p['probability']*math.log(p['probability'])for p in distribution if p['probability']);conditional=math.fsum(p['probability']*rec['entropy']for p,rec in zip(distribution,records))
    mix=dict(score=score,log_partition=-score/g,expected_base_cost=values['expected_cost'],expected_penalized_cost=math.fsum(p*v for p,v in zip(probabilities,shifted)),expected_length=mean,length_variance=variance,penalty_gradient=mean,penalty_curvature=-variance/g,temperature_gradient=values['temperature_gradient'],entropy=values['entropy'],length_entropy=entropy_length,conditional_entropy=conditional,distribution=distribution,occupancy=values['occupancy'],edges=values['edges'],gradient_x=values['gradient_x'],gradient_y=values['gradient_y'])
    return dict(schema=SCHEMA,request=req,alignment=oracle(r),lengths=records,penalized=mix)


def verify_lengths(report):
    if type(report)is not dict or 'request'not in report:raise ValueError('invalid length atlas')
    verify(report['alignment'])
    compare(length_oracle(report['request']),report)
    return True
