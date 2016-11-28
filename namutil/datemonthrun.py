from __future__ import print_function
import namutil
import sys
import os
import subprocess

dates = sys.argv[1]

for month in namutil.month_range(dates):
    args = [v.replace("__HDATE__", "{}-{}-01".format(month[:4], month[4:6])).replace("__DATE__", month + "01") for v in sys.argv[2:]]
    args = [v.replace("__HMONTH__", "{}-{}".format(month[:4], month[4:6])).replace("__MONTH__", month) for v in args]
    print(args)
    subprocess.check_call(args, stdout=sys.stdout, stderr=sys.stderr)

