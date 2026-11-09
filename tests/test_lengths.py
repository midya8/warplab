import copy
import math
import random
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from warplab.lengths import length_atlas
from warplab.length_oracle import verify_lengths
from warplab.core import kernel
from warplab.oracle import compare
R=dict(alignment=dict(x=[0,1,0],y=[0,0,1,0],gamma=.6),penalty=.5)

class LengthTests(unittest.TestCase):
    def test_complete_exhaustive_reports(self):
        rng=random.Random(483)
        for _ in range(100):
            n,m=rng.randrange(1,6),rng.randrange(1,6);r=dict(alignment=dict(x=[rng.randrange(-8,9)/4 for _ in range(n)],y=[rng.randrange(-8,9)/4 for _ in range(m)],gamma=rng.choice([.05,.2,1.,8.])),penalty=rng.choice([-4,-.5,0,.5,4]))
            with self.subTest(r=r):verify_lengths(length_atlas(r))

    def test_closed_form_length_counts_all_grids(self):
        for n in range(1,8):
            for m in range(1,8):
                r=length_atlas(dict(alignment=dict(x=[0]*n,y=[0]*m,gamma=1)))
                for s in r['lengths']:
                    ell=s['length'];down,right,diagonal=ell-m,ell-n,n+m-1-ell
                    count=math.factorial(ell-1)//(math.factorial(down)*math.factorial(right)*math.factorial(diagonal))
                    self.assertEqual(s['count'],count);self.assertEqual(s['hard']['ties'],count);self.assertAlmostEqual(s['score'],-math.log(count));self.assertAlmostEqual(sum(map(sum,s['occupancy'])),ell)
                self.assertEqual(sum(v['count']for v in r['lengths']),r['alignment']['hard']['total_paths'])

    def test_zero_penalty_recovers_base(self):
        for r in [R,dict(alignment=dict(x=[8]*7,y=[-8]*7,gamma=.05)),dict(alignment=dict(x=[0]*7,y=[0]*7,gamma=8)),dict(alignment=dict(x=[1],y=[0,1,2],gamma=.5))]:
            a=length_atlas(dict(r,penalty=0));s=a['alignment']['soft'];b=a['penalized']
            for key in ['score','log_partition','entropy','temperature_gradient','expected_length','occupancy','edges','gradient_x','gradient_y']:compare(s[key],b[key])
            verify_lengths(a)

    def test_conditional_conservation_entropy_chain(self):
        a=length_atlas(R)
        for s in a['lengths']:
            self.assertAlmostEqual(sum(map(sum,s['occupancy'])),s['length']);self.assertAlmostEqual(sum(e['probability']for e in s['edges']),s['length']-1)
            for i,row in enumerate(s['occupancy']):
                for j,v in enumerate(row):
                    if (i,j)!=(0,0):self.assertAlmostEqual(sum(e['probability']for e in s['edges']if e['target']==[i,j]),v)
                    if (i,j)!=(len(a['alignment']['cost'])-1,len(a['alignment']['cost'][0])-1):self.assertAlmostEqual(sum(e['probability']for e in s['edges']if e['source']==[i,j]),v)
        s=a['penalized'];self.assertAlmostEqual(s['entropy'],s['length_entropy']+s['conditional_entropy']);self.assertAlmostEqual(s['score'],s['expected_penalized_cost']-R['alignment']['gamma']*s['entropy'])

    def test_penalty_gradient_curvature_and_monotonic_length(self):
        eps=1e-4;a=length_atlas(R)['penalized'];lo=length_atlas(dict(R,penalty=R['penalty']-eps))['penalized'];hi=length_atlas(dict(R,penalty=R['penalty']+eps))['penalized']
        self.assertAlmostEqual((hi['score']-lo['score'])/(2*eps),a['penalty_gradient'],places=7);self.assertAlmostEqual((hi['expected_length']-lo['expected_length'])/(2*eps),a['penalty_curvature'],places=7)
        lengths=[length_atlas(dict(R,penalty=p))['penalized']['expected_length']for p in [-4,-2,0,2,4]];self.assertEqual(sorted(lengths,reverse=True),lengths)

    def test_penalty_agrees_with_independent_shifted_cost_kernel(self):
        a=length_atlas(R);r=a['alignment']['request'];c=a['alignment']['cost'];b=kernel([[v+R['penalty']for v in row]for row in c],r['gamma'],r['direction'])
        for key in ['score','occupancy','edges','entropy','temperature_gradient','expected_length']:compare(a['penalized'][key],b[key])

    def test_conditional_coordinate_and_temperature_finite_differences(self):
        eps=1e-5;a=length_atlas(R)
        for k in ['x','y']:
            for i in range(len(R['alignment'][k])):
                lo,hi=copy.deepcopy(R),copy.deepcopy(R);lo['alignment'][k][i]-=eps;hi['alignment'][k][i]+=eps;l,h=length_atlas(lo),length_atlas(hi)
                for s,sl,sh in zip(a['lengths'],l['lengths'],h['lengths']):self.assertAlmostEqual((sh['score']-sl['score'])/(2*eps),s['gradient_'+k][i],places=7)
                self.assertAlmostEqual((h['penalized']['score']-l['penalized']['score'])/(2*eps),a['penalized']['gradient_'+k][i],places=7)
        lo,hi=copy.deepcopy(R),copy.deepcopy(R);lo['alignment']['gamma']-=eps;hi['alignment']['gamma']+=eps
        for s,sl,sh in zip(a['lengths'],length_atlas(lo)['lengths'],length_atlas(hi)['lengths']):self.assertAlmostEqual((sh['score']-sl['score'])/(2*eps),s['temperature_gradient'],places=7)

    def test_independent_verifier_generator_disabled(self):
        a=length_atlas(R)
        with patch('warplab.lengths.length_atlas',side_effect=AssertionError),patch('warplab.lengths._messages',side_effect=AssertionError),patch('warplab.lengths._finish',side_effect=AssertionError),patch('warplab.core.kernel',side_effect=AssertionError):self.assertTrue(verify_lengths(a))

    def test_strict_inputs_and_forgeries(self):
        for r in [None,{},dict(R,extra=0),dict(R,penalty=True),dict(R,penalty=4.1),dict(R,penalty=float('nan'))]:
            with self.assertRaises(ValueError):length_atlas(r)
        for path,value in [(['schema'],'x'),(['penalized','expected_length'],1),(['penalized','length_entropy'],3),(['lengths',0,'count'],3.),(['lengths',0,'occupancy',0,0],False),(['lengths',1,'hard','ties'],99),(['lengths',0,'edges',0,'probability'],.83)]:
            a=length_atlas(R);at=a
            for k in path[:-1]:at=at[k]
            at[path[-1]]=value
            with self.subTest(path=path),self.assertRaises(ValueError):verify_lengths(a)

class LengthCLITests(unittest.TestCase):
    def call(self,args,body):return subprocess.run([sys.executable,'-m','warplab',*args],input=body,text=True,capture_output=True)
    def test_roundtrip(self):
        a=self.call(['lengths','-'],json.dumps(R));self.assertEqual(a.returncode,0,a.stderr);verify_lengths(json.loads(a.stdout))
        b=self.call(['verify-lengths','-'],a.stdout);self.assertEqual(b.returncode,0,b.stderr);self.assertTrue(json.loads(b.stdout)['verified'])
    def test_atomic_and_forged_output(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'atlas.json';p.write_text('preserve')
            a=self.call(['lengths','-','-o',str(p)],'{}');self.assertEqual(a.returncode,2);self.assertEqual(p.read_text(),'preserve')
            a=self.call(['lengths','-','-o',str(p)],json.dumps(R));self.assertEqual(a.returncode,0);report=json.loads(p.read_text());report['penalized']['expected_length']=99
            b=self.call(['verify-lengths','-'],json.dumps(report));self.assertEqual(b.returncode,2);self.assertEqual(b.stdout,'')
    def test_controls_and_report_size(self):
        controls=json.loads((Path(__file__).parents[1]/'docs/length-controls.json').read_text())
        for c in controls:
            a=self.call(['lengths','-'],json.dumps(c['input']));self.assertEqual(a.returncode,0,a.stderr);self.assertLess(len(a.stdout.encode()),4<<20)
            b=self.call(['verify-lengths','-'],a.stdout);self.assertEqual(b.returncode,0,b.stderr)

if __name__=='__main__':unittest.main()
