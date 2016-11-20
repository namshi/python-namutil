import namutil
import sys
import os
import subprocess

dates = sys.argv[1]
args = sys.argv[2:]

for month in namutil.month_range(dates):
    args = [v.replace("$HDATE", "{}-{}-01".format(month[:4], month[4:6])).replace("$DATE", month + "01") for v in args]
    args = [v.replace("$HMONTH", "{}-{}".format(month[:4], month[4:6])).replace("$MONTH", month) for v in args]
    subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

