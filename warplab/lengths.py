"""Length-resolved soft-DTW partitions and explicit visited-cell penalties."""
import math
from fractions import Fraction
from .core import analyze, number, request

SCHEMA = 'warplab-lengths-v1'


def length_request(data):
    if type(data)is not dict or set(data)-{'alignment','penalty'} or 'alignment' not in data:
        raise ValueError('expected alignment and optional penalty only')
    return dict(alignment=request(data['alignment']),penalty=number(data.get('penalty',0),-4,4))


def lse(xs):
    if not xs:return -math.inf
    top=max(xs)
    return top+math.log(math.fsum(math.exp(x-top)for x in xs))


def _messages(c,g,reverse=False):
    n,m=len(c),len(c[0]);size=n+m
    logs=[[[-math.inf]*size for _ in range(m)]for _ in range(n)]
    counts=[[[0]*size for _ in range(m)]for _ in range(n)]
    minima=[[[None]*size for _ in range(m)]for _ in range(n)]
    ties=[[[0]*size for _ in range(m)]for _ in range(n)]
    paths=[[[None]*size for _ in range(m)]for _ in range(n)]
    sign=1 if reverse else -1
    for ii in range(n):
        for jj in range(m):
            i=n-1-ii if reverse else ii;j=m-1-jj if reverse else jj
            ns=[(i+sign*a,j+sign*b)for a,b in ((0,1),(1,0),(1,1)) if 0<=i+sign*a<n and 0<=j+sign*b<m]
            if not ns:
                logs[i][j][1]=-c[i][j]/g;counts[i][j][1]=ties[i][j][1]=1;minima[i][j][1]=Fraction(c[i][j]);paths[i][j][1]=[[i,j]]
            else:
                for length in range(2,size):
                    valid=[(a,b)for a,b in ns if counts[a][b][length-1]]
                    if not valid:continue
                    logs[i][j][length]=-c[i][j]/g+lse([logs[a][b][length-1]for a,b in valid])
                    counts[i][j][length]=sum(counts[a][b][length-1]for a,b in valid)
                    best=min(minima[a][b][length-1]for a,b in valid)
                    winners=[(a,b)for a,b in valid if minima[a][b][length-1]==best]
                    minima[i][j][length]=best+Fraction(c[i][j]);ties[i][j][length]=sum(ties[a][b][length-1]for a,b in winners)
                    paths[i][j][length]=([[i,j]]+min(paths[a][b][length-1]for a,b in winners))if reverse else (min(paths[a][b][length-1]for a,b in winners)+[[i,j]])
    return logs,counts,minima,ties,paths


def _edges(n,m):
    return [(i,j,i+a,j+b)for i in range(n)for j in range(m)for a,b in ((0,1),(1,0),(1,1))if i+a<n and j+b<m]


def _gradients(cells,r):
    x,y=r['x'],r['y']
    return dict(gradient_x=[2*math.fsum(cells[i][j]*(a-b)for j,b in enumerate(y))for i,a in enumerate(x)],gradient_y=[2*math.fsum(cells[i][j]*(b-a)for i,a in enumerate(x))for j,b in enumerate(y)])


def length_atlas(data):
    req=length_request(data);r=req['alignment'];base=analyze(r);c=base['cost'];g=r['gamma'];n,m=len(c),len(c[0]);edges=_edges(n,m)
    f,counts,mins,ties,paths=_messages(c,g);b,*_=_messages(c,g,True)
    records=[]
    for length in range(max(n,m),n+m):
        z=f[-1][-1][length]
        occ=[]
        for i in range(n):
            row=[]
            for j in range(m):
                terms=[f[i][j][a]+b[i][j][length+1-a]+c[i][j]/g for a in range(1,length+1)if math.isfinite(f[i][j][a])and math.isfinite(b[i][j][length+1-a])]
                row.append(math.exp(min(0.,lse(terms)-z))if terms else 0.)
            occ.append(row)
        flows=[]
        for i,j,a,t in edges:
            terms=[f[i][j][k]+b[a][t][length-k]for k in range(1,length)if math.isfinite(f[i][j][k])and math.isfinite(b[a][t][length-k])]
            flows.append(dict(source=[i,j],target=[a,t],probability=math.exp(min(0.,lse(terms)-z))if terms else 0.))
        cost=math.fsum(occ[i][j]*c[i][j]for i in range(n)for j in range(m));score=-g*z;entropy=max(0.,(cost-score)/g)
        records.append(dict(length=length,count=counts[-1][-1][length],log_partition=z,score=score,expected_cost=cost,entropy=entropy,temperature_gradient=-entropy,occupancy=occ,edges=flows,hard=dict(score=float(mins[-1][-1][length]),ties=ties[-1][-1][length],path=paths[-1][-1][length]),**_gradients(occ,r)))
    return _finish(req,base,records)


def _finish(req,base,records):
    """Only the generator uses this mixture construction; verifier sums paths."""
    r=req['alignment'];g=r['gamma'];penalty=req['penalty'];n,m=len(r['x']),len(r['y'])
    logz=lse([v['log_partition']-penalty*v['length']/g for v in records]);probabilities=[math.exp(v['log_partition']-penalty*v['length']/g-logz)for v in records]
    distribution=[dict(length=v['length'],probability=p)for v,p in zip(records,probabilities)]
    mean=math.fsum(p*v['length']for p,v in zip(probabilities,records));variance=math.fsum(p*(v['length']-mean)**2 for p,v in zip(probabilities,records));basecost=math.fsum(p*v['expected_cost']for p,v in zip(probabilities,records));expected=basecost+penalty*mean
    entropy_length=-math.fsum(p*math.log(p)for p in probabilities if p);entropy_conditional=math.fsum(p*v['entropy']for p,v in zip(probabilities,records));entropy=entropy_length+entropy_conditional
    occ=[[math.fsum(p*v['occupancy'][i][j]for p,v in zip(probabilities,records))for j in range(m)]for i in range(n)]
    flows=[dict(source=e['source'],target=e['target'],probability=math.fsum(p*v['edges'][k]['probability']for p,v in zip(probabilities,records)))for k,e in enumerate(records[0]['edges'])]
    mix=dict(score=-g*logz,log_partition=logz,expected_base_cost=basecost,expected_penalized_cost=expected,expected_length=mean,length_variance=variance,penalty_gradient=mean,penalty_curvature=-variance/g,temperature_gradient=-entropy,entropy=entropy,length_entropy=entropy_length,conditional_entropy=entropy_conditional,distribution=distribution,occupancy=occ,edges=flows,**_gradients(occ,r))
    return dict(schema=SCHEMA,request=req,alignment=base,lengths=records,penalized=mix)
