"""Setup script for tournament policy server.

Scripted policies only need base cogames (no neural/GPU deps).
The server already has cogames installed, but we ensure the version matches.
"""
import subprocess
import sys
import shutil

uv = shutil.which("uv")
cmd = [uv, "pip", "install"] if uv else [sys.executable, "-m", "pip", "install"]
subprocess.check_call(cmd + ["cogames", "--quiet"])
