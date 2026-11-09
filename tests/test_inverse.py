import copy,json,math,random,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from warplab.inverse import solve_inverse,inverse_request,response,TOL
from warplab.inverse_oracle import verify_inverse
from warplab.lengths import length_atlas
CONTROLS=json.loads((Path(__file__).parents[1]/'docs/inverse-controls.json').read_text())

class InverseTests(unittest.TestCase):
 def test_all_controls(self):
  for c in CONTROLS:
   r=solve_inverse(c['input']);self.assertTrue(verify_inverse(r));self.assertLessEqual(len(r['trace']),40)
   if r['status']=='residual_tolerance':self.assertLessEqual(abs(r['residual']),TOL)
 def test_endpoint_and_unidentifiable(self):
  a=CONTROLS[0]['input']['alignment'];records=length_atlas({'alignment':a})['lengths']
  for p,status in [(-4,'lower_endpoint'),(4,'upper_endpoint')]:
   r=solve_inverse({'alignment':a,'target':response(records,1,p)['mean']});self.assertEqual(r['status'],status);self.assertTrue(verify_inverse(r))
  r=solve_inverse(CONTROLS[4]['input']);self.assertIsNone(r['selected']);self.assertIsNone(r['residual'])
 def test_verifier_does_not_call_generator(self):
  r=solve_inverse(CONTROLS[0]['input'])
  with patch('warplab.inverse.solve_inverse',side_effect=AssertionError('generator forbidden')),patch('warplab.inverse.length_atlas',side_effect=AssertionError('length DP forbidden')):self.assertTrue(verify_inverse(r))
 def test_numerically_flat_not_unique(self):
  r=solve_inverse(CONTROLS[7]['input']);self.assertEqual(r['status'],'numerically_flat');self.assertGreater(len(r['legal_lengths']),1);self.assertTrue(verify_inverse(r))
 def test_forgery(self):
  base=solve_inverse(CONTROLS[0]['input'])
  mutations=[lambda r:r.update(status='unidentifiable'),lambda r:r['trace'][0].update(lower=-3),lambda r:r['trace'][0].update(mean=99),lambda r:r.update(trace=r['trace'][:-1]),lambda r:r['selected'].update(derivative=1),lambda r:r.update(extra=True),lambda r:r['trace'].append(r['trace'][-1])]
  for mutation in mutations:
   r=copy.deepcopy(base);mutation(r)
   with self.assertRaises((ValueError,TypeError,KeyError)):verify_inverse(r)
 def test_invalid(self):
  a=CONTROLS[0]['input']
  for data in [None,{},dict(a,extra=1),dict(a,target=True),dict(a,target=0),dict(a,target=14),dict(a,target=float('nan')),dict(a,target=None)]:
   with self.assertRaises((ValueError,TypeError)):inverse_request(data)
 def test_random_interior_targets(self):
  rng=random.Random(732)
  for _ in range(20):
   a={'x':[rng.randint(-2,2)for _ in range(rng.randint(2,5))],'y':[rng.randint(-2,2)for _ in range(rng.randint(2,5))],'gamma':rng.choice([.2,.6,1,4,8])};records=length_atlas({'alignment':a})['lengths'];p=rng.uniform(-3,3);target=response(records,a['gamma'],p)['mean'];r=solve_inverse({'alignment':a,'target':target});self.assertTrue(verify_inverse(r))
   self.assertLess(abs(r['residual']),1e-7)
 def test_monotonicity_and_finite_difference(self):
  for c in CONTROLS:
   a=c['input']['alignment'];rec=length_atlas({'alignment':a})['lengths'];g=a['gamma'];old=math.inf
   for p in [-4,-2,0,2,4]:
    r=response(rec,g,p);self.assertLessEqual(r['mean'],old+1e-10);self.assertLessEqual(r['derivative'],0);old=r['mean']
    if abs(p)<4:
     eps=1e-5;fd=(response(rec,g,p+eps)['mean']-response(rec,g,p-eps)['mean'])/(2*eps);self.assertAlmostEqual(fd,r['derivative'],places=7)
 def test_cli_atomic_roundtrip(self):
  with tempfile.TemporaryDirectory()as d:
   f=Path(d)/'in.json';o=Path(d)/'out.json';f.write_text(json.dumps(CONTROLS[0]['input']));subprocess.run([sys.executable,'-m','warplab','inverse',str(f),'-o',str(o)],check=True);subprocess.run([sys.executable,'-m','warplab','verify-inverse',str(o)],check=True,capture_output=True);old=o.read_bytes();f.write_text('{}');p=subprocess.run([sys.executable,'-m','warplab','inverse',str(f),'-o',str(o)],capture_output=True);self.assertEqual(p.returncode,2);self.assertEqual(old,o.read_bytes())
