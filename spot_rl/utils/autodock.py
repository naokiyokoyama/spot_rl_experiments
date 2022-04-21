import subprocess
import sys

subprocess.check_call(f"{sys.executable} spot_rl.envs.nav_env -w dock -d", shell=True)
