"""Independent exhaustive path oracle. No dynamic-programming calls."""
import math
from fractions import Fraction
from .core import request, VERSION


def enumerate_paths(n,m):
    stack = [[[0,0]]]
    while stack:
        p = stack.pop()
        i,j = p[-1]
        if i==n-1 and j==m-1:
            yield p
        else:
            for a,b in ((1,1),(1,0),(0,1)):
                if i+a<n and j+b<m:
                    stack.append(p+[[i+a,j+b]])


def oracle(data):
    r=request(data);x,y,g,d=(r[k] for k in ('x','y','gamma','direction'));n,m=len(x),len(y)
    c=[[(a-b)**2 for b in y] for a in x]
    paths=list(enumerate_paths(n,m))
    exact=[sum((Fraction(c[i][j]) for i,j in p),Fraction(0)) for p in paths]
    costs=list(map(float,exact))
    directions=[sum(d[i][j] for i,j in p) for p in paths]
    minimum=min(costs)
    weights=[math.exp(-(cost-minimum)/g) for cost in costs];z=math.fsum(weights);probs=[w/z for w in weights]
    score=minimum-g*math.log(z)
    expected=math.fsum(p*a for p,a in zip(probs,costs));mean=math.fsum(p*a for p,a in zip(probs,directions))
    entropy=-math.fsum(p*math.log(p) for p in probs if p)
    occ=[[math.fsum(prob for path,prob in zip(paths,probs) if [i,j] in path) for j in range(m)] for i in range(n)]
    hvp=[[-math.fsum(prob*(v-mean) for path,prob,v in zip(paths,probs,directions) if [i,j] in path)/g for j in range(m)] for i in range(n)]
    edges=[]
    for i in range(n):
        for j in range(m):
            for a,b in ((0,1),(1,0),(1,1)):
                if i+a<n and j+b<m:
                    edge=([i,j],[i+a,j+b]);p=math.fsum(prob for path,prob in zip(paths,probs) if any(tuple(path[k:k+2])==edge for k in range(len(path)-1)))
                    edges.append(dict(source=[i,j],target=[i+a,j+b],probability=p))
    exact_min=min(exact)
    winners=[p for p,v in zip(paths,exact) if v==exact_min]
    soft=dict(score=score,log_partition=-score/g,expected_cost=expected,entropy=entropy,temperature_gradient=-entropy,expected_length=math.fsum(p*len(path) for p,path in zip(probs,paths)),occupancy=occ,edges=edges,cost_hvp=hvp,directional_mean=mean,directional_curvature=-math.fsum(p*(v-mean)**2 for p,v in zip(probs,directions))/g,gradient_x=[2*math.fsum(occ[i][j]*(a-b) for j,b in enumerate(y))for i,a in enumerate(x)],gradient_y=[2*math.fsum(occ[i][j]*(b-a) for i,a in enumerate(x))for j,b in enumerate(y)])
    return dict(schema=VERSION,request=r,cost=c,hard=dict(score=minimum,ties=len(winners),path=min(winners),total_paths=len(paths)),soft=soft)


def compare(a,b,path='root'):
    if type(a) is int:
        if type(b) is not int or a!=b:raise ValueError('integer mismatch at '+path)
    elif type(a) in (int,float) and type(b) in (int,float):
        if not math.isfinite(b) or abs(a-b)>2e-7*max(1,abs(a),abs(b)):
            raise ValueError('numeric mismatch at '+path)
    elif type(a) is not type(b):
        raise ValueError('type mismatch at '+path)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise ValueError('keys mismatch at '+path)
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('length mismatch at '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
    elif a!=b:raise ValueError('value mismatch at '+path)


def verify(report):
    if type(report) is not dict or 'request' not in report:raise ValueError('invalid report')
    compare(oracle(report['request']),report)
    # Integer counts, coordinates and schema cannot be weakened to floats.
    if type(report['hard']['ties'])is not int or type(report['hard']['total_paths'])is not int:raise ValueError('integer counts required')
    for cell in report['hard']['path']:
        if any(type(v)is not int for v in cell):raise ValueError('integer coordinates required')
    return True
