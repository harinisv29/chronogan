import pytest
from pathlib import Path
root=Path(__file__).parent
(root/'results'/'gpu').mkdir(exist_ok=True)
raise SystemExit(pytest.main(['tests','-q','--basetemp',str(root/'results'/'pytest-tmp'),'--junitxml',str(root/'results'/'gpu'/'tests.xml')]))
