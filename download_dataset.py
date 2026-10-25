"""Download and validate the original PhysioNet archives; never synthesize data."""
from pathlib import Path
import json
import shutil
import zipfile
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
BASE = 'https://physionet.org/files/challenge-2019/1.0.0/'

class PatientLinks(HTMLParser):
    def __init__(self): super().__init__(); self.files=[]
    def handle_starttag(self,tag,attrs):
        href=dict(attrs).get('href','')
        if tag=='a' and href.endswith('.psv') and '/' not in href and '\\' not in href:
            self.files.append(href)

def download_directory(name,limit=0):
    """Official directory fallback: the old ZIP endpoints may be unavailable."""
    group=name.removesuffix('.zip'); url=BASE+'training/'+group+'/'
    listing=requests.get(url,timeout=(15,120)); listing.raise_for_status()
    parser=PatientLinks(); parser.feed(listing.text)
    names=sorted(set(parser.files))
    expected=20336 if group.endswith('A') else 20000
    if len(names)!=expected: raise ValueError(f'{url}: expected {expected} patient links, got {len(names)}')
    if limit: names=names[:limit]
    destination=DATA/group; destination.mkdir(exist_ok=True)
    local=threading.local()
    def fetch(filename):
        path=destination/filename
        if path.exists() and path.stat().st_size>0:
            with path.open() as source:
                header=source.readline()
            if header.startswith('HR|') and 'SepsisLabel' in header: return path
        if not hasattr(local,'session'):
            local.session=requests.Session()
            local.session.mount('https://',HTTPAdapter(max_retries=Retry(total=4,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
        response=local.session.get(url+filename,timeout=(15,120)); response.raise_for_status()
        if not response.text.startswith('HR|') or 'SepsisLabel' not in response.text.splitlines()[0]:
            raise ValueError(f'Invalid patient response: {filename}')
        temp=path.with_suffix('.part'); temp.write_bytes(response.content); temp.replace(path)
        return path
    with ThreadPoolExecutor(max_workers=12) as pool:
        paths=list(tqdm(pool.map(fetch,names),total=len(names),desc=f'Official {group} records',file=sys.stdout))
    return paths

def valid_zip(path):
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and any(n.endswith('.psv') for n in z.namelist())
    except (OSError, zipfile.BadZipFile):
        return False

def download(name,limit=0):
    DATA.mkdir(exist_ok=True)
    archive = DATA / name
    if not valid_zip(archive):
        session = requests.Session()
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=[429,500,502,503,504])))
        partial = archive.with_suffix('.part')
        with session.get(BASE + name, stream=True, timeout=(15,120)) as response:
            if response.status_code==404:
                print(f'{name}: old ZIP URL unavailable; using official patient directory',flush=True)
                return download_directory(name,limit)
            response.raise_for_status()
            with partial.open('wb') as out, tqdm(total=int(response.headers.get('Content-Length',0)), unit='B', unit_scale=True, desc=name,file=sys.stdout) as bar:
                for chunk in response.iter_content(1024*1024):
                    out.write(chunk)
                    bar.update(len(chunk))
        if not valid_zip(partial):
            raise ValueError(f'Corrupt archive: {partial}')
        partial.replace(archive)
    destination = DATA / name.removesuffix('.zip')
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist() if n.endswith('.psv')]
        if len({Path(n).name for n in members}) != len(members):
            raise ValueError('Duplicate patient names in archive')
        for member in tqdm(members, desc='Extracting',file=sys.stdout):
            target = destination / Path(member).name
            if not target.exists() or target.stat().st_size != z.getinfo(member).file_size:
                with z.open(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src,dst)
    paths=sorted(destination.glob('*.psv'))
    return paths[:limit] if limit else paths

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--limit-per-set',type=int,default=0,help='Real-data smoke subset; zero downloads all patients')
    args=parser.parse_args()
    if args.limit_per_set<0: parser.error('Limit cannot be negative')
    groups = {s: download(f'training_set{s}.zip',args.limit_per_set) for s in ('A','B')}
    total = sum(map(len, groups.values()))
    if not args.limit_per_set and total != 40336:
        raise ValueError(f'Expected 40336 patients, found {total}')
    rows = 0
    missing = None
    for files in groups.values():
        for path in tqdm(files, desc='Summarizing real records',file=sys.stdout):
            frame = pd.read_csv(path, sep='|')
            rows += len(frame)
            missing = frame.isna().sum() if missing is None else missing.add(frame.isna().sum(), fill_value=0)
    summary = {'source': BASE, 'subset':bool(args.limit_per_set), 'expected_full_patients':40336,'patients': total, 'sets': {s:len(f) for s,f in groups.items()}, 'observations': rows,
               'variables': list(missing.index), 'missingness_percent': (100*missing/rows).to_dict()}
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
