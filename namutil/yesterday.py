import sys
import datetime

yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
sys.stdout.write(yesterday.strftime("%Y%m%d"))
