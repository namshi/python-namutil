import sys
import os
import jinja2

sys.stdout.write(jinja2.Template(sys.stdin.read()).render(**os.environ))

