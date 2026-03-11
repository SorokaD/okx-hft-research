"""
Add project root to sys.path so 'research' can be imported without pip install.
Run first in notebooks: %run bootstrap.py
Or: exec(open("bootstrap.py").read())  # when cwd is project root
"""
import sys
from pathlib import Path

def _find_root():
    for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
        if (p / "pyproject.toml").exists() and (p / "research").is_dir():
            return p
    return Path.cwd()

_root = _find_root()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
