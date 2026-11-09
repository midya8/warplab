"""Run outside checkout with a freshly installed wheel."""
import json,subprocess,sys,tempfile
from pathlib import Path
from warplab.inverse_oracle import verify_inverse
cs=json.loads(Path(sys.argv[1]).read_text());console=sys.argv[2]
with tempfile.TemporaryDirectory()as d:
 for c in cs:
  f=Path(d)/'in.json';o=Path(d)/'out.json';f.write_text(json.dumps(c['input']))
  for cmd in ([sys.executable,'-m','warplab'],[console]):
   subprocess.run([*cmd,'inverse',str(f),'-o',str(o)],check=True);assert verify_inverse(json.loads(o.read_text()));assert json.loads(subprocess.check_output([*cmd,'verify-inverse',str(o)],text=True))['verified']
print('Installed inverse module and console: all %s controls independently verified' % len(cs))
