import sys
from os.path import dirname, join, abspath
minspan_dir = abspath(join(dirname(__file__), "..", ".."))
sys.path.insert(0, minspan_dir)
del sys, dirname, join, abspath
del minspan_dir
