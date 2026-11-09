"""Run outside the checkout, using only the freshly installed wheel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from warplab import verify
controls=json.loads(Path(sys.argv[1]).read_text());console=sys.argv[2]
with tempfile.TemporaryDirectory()as d:
    for c in controls:
        request=Path(d)/'request.json';output=Path(d)/'certificate.json';request.write_text(json.dumps(c['input']))
        for cmd in ([sys.executable,'-m','warplab'],[console]):
            subprocess.run([*cmd,'analyze',str(request),'-o',str(output)],check=True)
            report=json.loads(output.read_text());assert verify(report)
            result=json.loads(subprocess.check_output([*cmd,'verify',str(output)],text=True));assert result['verified']
print('Installed module and console: all %s controls verified independently' % len(controls))
