"""Make `src` importable when running pytest from the repo root."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(_HERE, "..", ".."))
if os.path.join(_SRC, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_SRC, "src"))
