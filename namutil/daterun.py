import namutil
import sys
import os
import subprocess

dates = sys.argv[1]

for date in namutil.date_range(dates):
    args = [v.replace("__HDATE__", "{}-{}-{}".format(date[:4], date[4:6], date[6:])).replace("__DATE__", date) for v in sys.argv[2:]]
    print(args)
    subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

