import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from warplab import analyze,verify
from warplab.core import kernel
from warplab.oracle import oracle,compare

R=dict(x=[0,1,0],y=[0,0,1,0],gamma=.6)

class CoreTests(unittest.TestCase):
    def test_exhaustive_dyadic_cases(self):
        rng=random.Random(713)
        for _ in range(140):
            n,m=rng.randrange(1,6),rng.randrange(1,6)
            r=dict(x=[rng.randrange(-8,9)/4 for _ in range(n)],y=[rng.randrange(-8,9)/4 for _ in range(m)],gamma=rng.choice([.05,.2,1.,8.]),direction=[[rng.randrange(-4,5)for _ in range(m)]for _ in range(n)])
            with self.subTest(r=r):verify(analyze(r))

    def test_extreme_and_largest_path_space(self):
        for r in [dict(x=[0]*7,y=[0]*7,gamma=8),dict(x=[8]*7,y=[-8]*7,gamma=.05),dict(x=[8],y=[-8]*7,gamma=.05),dict(x=[.1,.2,.3],y=[.3,.2,.1],gamma=.05)]:verify(analyze(r))
        a=analyze(dict(x=[0]*7,y=[0]*7,gamma=1));self.assertEqual(a['hard']['total_paths'],8989);self.assertEqual(a['hard']['ties'],8989)
        self.assertAlmostEqual(a['soft']['score'],-math.log(8989));self.assertAlmostEqual(a['soft']['entropy'],math.log(8989))

    def test_independent_oracle_no_generator(self):
        a=analyze(R)
        with patch('warplab.core.analyze',side_effect=AssertionError),patch('warplab.core.kernel',side_effect=AssertionError),patch('warplab.core.hard',side_effect=AssertionError):self.assertTrue(verify(a))

    def test_cell_and_edge_conservation(self):
        a=analyze(R);s=a['soft'];n,m=len(R['x']),len(R['y'])
        self.assertAlmostEqual(s['occupancy'][0][0],1);self.assertAlmostEqual(s['occupancy'][-1][-1],1)
        for i in range(n):
            for j in range(m):
                for k,exception in [('source',(n-1,m-1)),('target',(0,0))]:
                    if (i,j)!=exception:self.assertAlmostEqual(sum(e['probability']for e in s['edges']if e[k]==[i,j]),s['occupancy'][i][j])
        self.assertAlmostEqual(sum(e['probability']for e in s['edges']),s['expected_length']-1)

    def test_coordinate_gradients_finite_difference(self):
        a=analyze(R);eps=1e-5
        for k in ['x','y']:
            for i in range(len(R[k])):
                lo,hi=copy.deepcopy(R),copy.deepcopy(R);lo[k][i]-=eps;hi[k][i]+=eps
                fd=(analyze(hi)['soft']['score']-analyze(lo)['soft']['score'])/(2*eps)
                self.assertAlmostEqual(fd,a['soft']['gradient_'+k][i],places=7)

    def test_temperature_gradient_and_variance(self):
        a=analyze(R);eps=1e-5
        lo,hi=dict(R,gamma=R['gamma']-eps),dict(R,gamma=R['gamma']+eps)
        self.assertAlmostEqual((analyze(hi)['soft']['score']-analyze(lo)['soft']['score'])/(2*eps),a['soft']['temperature_gradient'],places=7)
        self.assertLessEqual(a['soft']['temperature_gradient'],0)

    def test_cost_hvp_finite_difference_and_symmetry(self):
        a=analyze(R);c=a['cost'];g=R['gamma'];d=a['request']['direction'];e=[[.1*(i+2*j)for j in range(len(c[0]))]for i in range(len(c))];eps=1e-5
        hi=kernel([[v+eps*d[i][j]for j,v in enumerate(row)]for i,row in enumerate(c)],g,d)
        lo=kernel([[v-eps*d[i][j]for j,v in enumerate(row)]for i,row in enumerate(c)],g,d)
        for i in range(len(c)):
            for j in range(len(c[0])):self.assertAlmostEqual((hi['occupancy'][i][j]-lo['occupancy'][i][j])/(2*eps),a['soft']['cost_hvp'][i][j],places=7)
        he=kernel(c,g,e)['cost_hvp'];hd=a['soft']['cost_hvp']
        self.assertAlmostEqual(sum(e[i][j]*hd[i][j]for i in range(len(c))for j in range(len(c[0]))),sum(d[i][j]*he[i][j]for i in range(len(c))for j in range(len(c[0]))))
        self.assertLessEqual(a['soft']['directional_curvature'],1e-9)

    def test_bounds_transpose_and_scaling(self):
        a=analyze(R);b=analyze(dict(x=R['y'],y=R['x'],gamma=R['gamma']));s=a['soft'];h=a['hard']
        self.assertAlmostEqual(s['score'],b['soft']['score']);self.assertLessEqual(s['score'],h['score']+1e-9);self.assertGreaterEqual(s['score'],h['score']-R['gamma']*math.log(h['total_paths'])-1e-9)
        for i,row in enumerate(s['occupancy']):
            for j,v in enumerate(row):self.assertAlmostEqual(v,b['soft']['occupancy'][j][i])
        c=a['cost'];d=a['request']['direction'];t=kernel([[v*2 for v in row]for row in c],2*R['gamma'],d)
        self.assertAlmostEqual(t['score'],2*s['score']);compare(t['occupancy'],s['occupancy'])

    def test_single_path(self):
        a=analyze(dict(x=[1],y=[0,1,2],gamma=.5));self.assertEqual(a['hard']['total_paths'],1);self.assertEqual(a['soft']['entropy'],0);self.assertEqual(a['soft']['cost_hvp'],[[0.,0.,0.]])

    def test_forgery_rejected(self):
        a=analyze(R)
        for path,value in [(['soft','score'],12),(['soft','entropy'],4),(['soft','cost_hvp',0,0],8),(['soft','occupancy',0,0],False),(['hard','ties'],1.),(['hard','path',0,0],0.),(['soft','edges',0,'probability'],.77),(['request','gamma'],True),(['schema'],'x')]:
            b=copy.deepcopy(a);at=b
            for k in path[:-1]:at=at[k]
            at[path[-1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):verify(b)
        b=copy.deepcopy(a);b['extra']=0
        with self.assertRaises(ValueError):verify(b)

    def test_bad_requests(self):
        for r in [None,{},dict(R,z=0),dict(R,x=[]),dict(R,x=[0]*8),dict(R,y=[True]),dict(R,gamma=0),dict(R,gamma=float('nan')),dict(R,x=[9]),dict(R,direction=[[0]])]:
            with self.subTest(r=r),self.assertRaises(ValueError):analyze(r)

    def test_input_not_mutated(self):
        r=copy.deepcopy(R);analyze(r);self.assertEqual(r,R)

class CLITests(unittest.TestCase):
    def call(self,args,body):return subprocess.run([sys.executable,'-m','warplab',*args],input=body,text=True,capture_output=True)
    def test_roundtrip(self):
        a=self.call(['analyze','-'],json.dumps(R));self.assertEqual(a.returncode,0,a.stderr)
        b=self.call(['verify','-'],a.stdout);self.assertEqual(b.returncode,0,b.stderr);self.assertTrue(json.loads(b.stdout)['verified'])
    def test_duplicate_nonfinite_oversize(self):
        for raw in ['{"x":[0],"x":[1],"y":[0],"gamma":1}','{"x":[NaN],"y":[0],"gamma":1}','x'*(4*1024*1024+1)]:
            a=self.call(['analyze','-'],raw);self.assertEqual(a.returncode,2);self.assertEqual(a.stdout,'')
    def test_atomic_output(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'report.json';p.write_text('preserve')
            a=self.call(['analyze','-','-o',str(p)],'{}');self.assertEqual(a.returncode,2);self.assertEqual(p.read_text(),'preserve')
            a=self.call(['analyze','-','-o',str(p)],json.dumps(R));self.assertEqual(a.returncode,0);verify(json.loads(p.read_text()));self.assertEqual(len(list(Path(d).iterdir())),1)

if __name__=='__main__':unittest.main()
