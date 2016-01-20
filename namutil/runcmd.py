import UserDict
class ChainMap(UserDict.DictMixin):
    def __init__(self, *maps): self._maps = maps
    def __getitem__(self, key):
        for mapping in self._maps:
            try: return mapping[key]
            except KeyError: pass
        raise KeyError(key)

def run(c, input=None, **kwargs):
    import subprocess
    p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE if input else None)
    ret = p.communicate(input=input)
    assert p.returncode == 0, ret
    return ret

def pr(c):
    import sys, string
    l = sys._getframe(1).f_locals
    print string.Formatter().vformat(c, [], ChainMap(sys._getframe(1).f_locals, sys._getframe(1).f_globals))

def cmd(c, **kwargs):
    import sys, string
    nc = string.Formatter().vformat(c, [], ChainMap(sys._getframe(1).f_locals, sys._getframe(1).f_globals))
    return run(nc, **kwargs)

def pcmd(c, **kwargs):
    import sys, string
    nc = string.Formatter().vformat(c, [], ChainMap(sys._getframe(1).f_locals, sys._getframe(1).f_globals))
    ret = run(nc, **kwargs)
    print ret[0]
    print ret[1]

