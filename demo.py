import json
from pathlib import Path
from spectral import analyze
from score import transcribe

cases=json.loads((Path(__file__).parent/'fixtures/cases.json').read_text())
spectral=analyze(**cases['ordinary_spectral'])
notes=transcribe(spectral['activations'],**cases['ordinary_score'])
print(json.dumps({'spectral':spectral,'score':notes},indent=2))
