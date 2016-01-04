import namutil
import sys
import os
import subprocess

dates = sys.argv[1]

for date in namutil.date_range(dates):
    args = [v.replace("$DATE", date) for v in sys.argv[2:]]
    subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

