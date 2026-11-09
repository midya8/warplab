"""Scalar soft-DTW on a bounded, fully inspectable monotone path space."""
import math
from fractions import Fraction

VERSION = 'warplab-v1'
MOVES = ((0, 1), (1, 0), (1, 1))


def number(v, lo, hi):
    if type(v) not in (int, float) or not lo <= v <= hi or not math.isfinite(v):
        raise ValueError('expected a finite number in [%s, %s]' % (lo, hi))
    return float(v)


def request(data):
    if type(data) is not dict or not {'x', 'y', 'gamma'} <= data.keys() or data.keys() - {'x', 'y', 'gamma', 'direction'}:
        raise ValueError('expected x, y, gamma and optional direction only')
    for k in ('x', 'y'):
        if type(data[k]) is not list or not 1 <= len(data[k]) <= 7:
            raise ValueError('sequence lengths must be 1..7')
    x = [number(v, -8, 8) for v in data['x']]
    y = [number(v, -8, 8) for v in data['y']]
    g = number(data['gamma'], .05, 8)
    d = data.get('direction', [[(-1.) ** (i + j) for j in range(len(y))] for i in range(len(x))])
    if type(d) is not list or len(d) != len(x) or any(type(row) is not list or len(row) != len(y) for row in d):
        raise ValueError('direction must have shape len(x) by len(y)')
    return dict(x=x, y=y, gamma=g, direction=[[number(v, -4, 4) for v in row] for row in d])


def _mix(logs, means):
    top = max(logs)
    weights = [math.exp(v - top) for v in logs]
    z = math.fsum(weights)
    return top + math.log(z), math.fsum(w * v for w, v in zip(weights, means)) / z


def kernel(c, g, d):
    """Log-partition and directional expectation messages, O(n*m)."""
    n, m = len(c), len(c[0])
    f, b, u, v = [[[0.] * m for _ in range(n)] for _ in range(4)]
    for rows, cols, msg, avg, sign in ((range(n), range(m), f, u, -1), (range(n-1, -1, -1), range(m-1, -1, -1), b, v, 1)):
        for i in rows:
            for j in cols:
                neighbors = [(i + sign * a, j + sign * z) for a, z in MOVES if 0 <= i + sign * a < n and 0 <= j + sign * z < m]
                total, mean = _mix([msg[a][z] for a, z in neighbors], [avg[a][z] for a, z in neighbors]) if neighbors else (0., 0.)
                msg[i][j] = total - c[i][j] / g
                avg[i][j] = mean + d[i][j]
    z = f[-1][-1]
    occupancy = [[math.exp(min(0., f[i][j] + b[i][j] + c[i][j]/g - z)) for j in range(m)] for i in range(n)]
    hvp = [[-occupancy[i][j] * (u[i][j] + v[i][j] - d[i][j] - u[-1][-1]) / g for j in range(m)] for i in range(n)]
    edges = []
    for i in range(n):
        for j in range(m):
            for a, t in MOVES:
                if i+a < n and j+t < m:
                    edges.append(dict(source=[i,j], target=[i+a,j+t], probability=math.exp(min(0., f[i][j]+b[i+a][j+t]-z))))
    expected = math.fsum(occupancy[i][j]*c[i][j] for i in range(n) for j in range(m))
    score = -g*z
    entropy = max(0., (expected-score)/g)
    return dict(score=score, log_partition=z, expected_cost=expected, entropy=entropy, temperature_gradient=-entropy, expected_length=math.fsum(map(math.fsum,occupancy)), occupancy=occupancy, edges=edges, cost_hvp=hvp, directional_mean=u[-1][-1], directional_curvature=math.fsum(d[i][j]*hvp[i][j] for i in range(n) for j in range(m)))


def hard(c):
    n, m = len(c), len(c[0])
    score = [[Fraction(0)] * m for _ in range(n)]
    count = [[0] * m for _ in range(n)]
    paths = [[[] for _ in range(m)] for _ in range(n)]
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):
            ns = [(i+a,j+b) for a,b in MOVES if i+a<n and j+b<m]
            best = min((score[a][b] for a,b in ns),default=Fraction(0))
            tied = [(a,b) for a,b in ns if score[a][b] == best]
            score[i][j] = Fraction(c[i][j])+best
            count[i][j] = sum(count[a][b] for a,b in tied) if tied else 1
            paths[i][j] = [[i,j]] + (min(paths[a][b] for a,b in tied) if tied else [])
    # Counts of all paths use an independent integer recurrence.
    total = [[0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            total[i][j] = sum(total[i-a][j-b] for a,b in MOVES if i>=a and j>=b) if i or j else 1
    return dict(score=float(score[0][0]), ties=count[0][0], path=paths[0][0], total_paths=total[-1][-1])


def analyze(data):
    r = request(data)
    x,y,g,d = (r[k] for k in ('x','y','gamma','direction'))
    c = [[(a-b)**2 for b in y] for a in x]
    result = kernel(c,g,d)
    result['gradient_x'] = [2*math.fsum(result['occupancy'][i][j]*(a-b) for j,b in enumerate(y)) for i,a in enumerate(x)]
    result['gradient_y'] = [2*math.fsum(result['occupancy'][i][j]*(b-a) for i,a in enumerate(x)) for j,b in enumerate(y)]
    return dict(schema=VERSION, request=r, cost=c, hard=hard(c), soft=result)
