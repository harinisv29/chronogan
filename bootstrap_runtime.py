"""Copy a minimal local embedded runtime; never change the source installation."""
from pathlib import Path
import shutil
import sys

root=Path(__file__).resolve().parent
source=Path(sys.executable).parent
target=root/'.runtime'/'python'
target.mkdir(parents=True,exist_ok=True)
for item in source.iterdir():
    if item.is_file() and (item.suffix.lower() in ['.exe','.dll','.pyd','.zip'] or item.name=='LICENSE.txt'):
        destination=target/item.name
        if not destination.exists(): shutil.copy2(item,destination)
pip_source=source/'Lib'/'site-packages'/'pip'
pip_target=target/'Lib'/'site-packages'/'pip'
if not pip_target.exists(): shutil.copytree(pip_source,pip_target)
(target/'python312._pth').write_text('python312.zip\n.\nLib\\site-packages\nimport site\n')
print(f'ChronoGAN-only interpreter: {target / "python.exe"}')
