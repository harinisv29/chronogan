import zipfile
from download_dataset import valid_zip,PatientLinks

def test_archive_integrity(tmp_path):
    path=tmp_path/'patients.zip'
    assert not valid_zip(path)
    with zipfile.ZipFile(path,'w') as z: z.writestr('training/p1.psv','HR|ICULOS\n80|1\n')
    assert valid_zip(path)
    path.write_bytes(b'broken zip')
    assert not valid_zip(path)

def test_official_directory_links():
    parser=PatientLinks()
    parser.feed('<a href="p000001.psv">patient</a><a href="../">parent</a><a href="../bad.psv">bad</a>')
    assert parser.files==['p000001.psv']
