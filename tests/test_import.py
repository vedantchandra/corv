import subprocess
import sys

def test_import_is_silent():
    out = subprocess.run([sys.executable, '-c', 'import corv'], capture_output = True, text = True)
    assert out.returncode == 0
    assert out.stdout == '' and out.stderr == ''
