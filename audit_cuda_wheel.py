"""Independently verify the downloaded wheel against the official index SHA-256."""
from pathlib import Path
import json
import tempfile
from install_gpu import ROOT,NAME,SHA256,INDEX,digest

candidates=list((ROOT/'.gpu-download').rglob(NAME))
candidates.extend(p/NAME for p in Path(tempfile.gettempdir()).glob('pip-unpack-*'))
candidates.extend(p for p in (ROOT/'.pip-cache').rglob('*.body') if p.stat().st_size==1915224818)
for path in candidates:
    if path.exists() and path.stat().st_size==1915224818:
        actual=digest(path)
        if actual!=SHA256: raise SystemExit(f'CUDA wheel checksum mismatch: {path}')
        report={'source_index':INDEX,'wheel':NAME,'size':path.stat().st_size,'sha256':actual,'expected_sha256':SHA256,'verified':True,'cached_file':str(path)}
        (ROOT/'results'/'gpu'/'wheel-integrity.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2)); break
else: raise SystemExit('No complete cached CUDA wheel found to audit')
