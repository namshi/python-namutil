from __future__ import print_function

try:
    basestring
except NameError:
    basestring = str

try:
    range = xrange
except NameError:
    pass


def get_handler(level='INFO'):
    import logging
    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(logging.Formatter('%(asctime)-15s %(levelname)s %(message)s'))
    return h

def get_logger(facility, **kwargs):
    import logging
    configure_logging(**kwargs)
    return logging.getLogger(facility)

def configure_logging(level='INFO', format='%(asctime)-15s %(levelname)s %(message)s'):
    import logging
    logging.basicConfig(
        format=format,
        level=level
    )

def get_graylogger(host, facility, level='INFO', port=12201, **kwargs):
    import logging, graypy
    logger = logging.getLogger(facility)
    logger.setLevel(getattr(logging, level))
    logger.addHandler(graypy.GELFHandler(host, port, **kwargs))
    h = logging.StreamHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(h)
    logger.info("Starting")
    return logger

def get_beater(group, process, key, sleep="10m", logger=None):
    import requests
    def beat():
        try:
            ret = requests.get("https://ht-beat.appspot.com/v1/heartbeat/{group}/{process}/?sleep={sleep}&key={key}".format(group=group, process=process, key=key, sleep=sleep))
            assert ret.status_code == 200, "Status code %s" % ret.status_code
        except Exception as e:
            if logger:
                logger.warn("Heartbeat failed! %s" % str(e))
            else:
                print("Heartbeat failed! %s" % str(e))
    return beat

sns_conn = None
def get_sns_conn():
    from boto import sns
    global sns_conn
    if sns_conn is None:
        sns_conn = sns.connect_to_region("ap-southeast-1")
    return sns_conn

def get_old_google_doc_export_url(key, gid='0'):
    import urlparse
    url = urlparse.urlparse(key.replace('#', '&'))
    if url.query:
        qs = urlparse.parse_qs(url.query)
        key, gid = qs['key'][0], qs.get('gid', ['0'])[0]
    return "https://docs.google.com/spreadsheet/pub?key=%s&single=true&gid=%s&output=txt" % (key, gid)

def get_google_doc_export_url(url, *args):
    if '/spreadsheets/d/' not in url:
        return get_old_google_doc_export_url(url, *args)
    return url.replace('/edit#gid=', '/export?format=tsv&gid=')

def read_google_docs(docs):
    header = None
    docs = [docs] if isinstance(docs, basestring) else docs
    for doc in docs:
        doc_header = None
        for row in read_google_doc(doc):
            if header is None: header = set(row.keys())
            if doc_header is None:
                doc_header = set(row.keys())
                assert header == doc_header, "different headers on '{}'! \n{}\n{}\n".format(doc, repr(header), repr(doc_header))
            yield row

def read_google_doc(*args, **kwargs):
    try: from urllib import request
    except ImportError: import urllib2 as request
    export_url = get_google_doc_export_url(*args)
    resp = request.urlopen(export_url)
    assert resp.headers['content-type'] == 'text/tab-separated-values', "bad content type '{}'; ensure that sheet is published: {}".format(resp.headers['content-type'], repr(args))
    data = resp.read().decode('utf8')
    header, rows = data.split("\n", 1)
    header = header.split("\t")
    rows = rows.replace("\n", "\t").split("\t")
    rows = list(grouper(len(header), rows))
    dictc = kwargs.get('dict', dict)
    sanitize = (lambda v: v.strip().replace("\r\n", "\n").replace("\r", "\n")) if kwargs.get('strip', True) else (lambda v: v.replace("\r\n", "\n").replace("\r", "\n"))
    is_valid_key = lambda k: bool(k and k.strip() and k[0] != '-')
    return [dictc((sanitize(k), sanitize(v)) for k, v in zip(header, row) if is_valid_key(k)) for row in rows]

class memoize(object):
   def __init__(self, expiry_time=0, cache=None, num_args=None, locked=False):
       import threading
       self.cache = {} if cache is None else cache
       self.expiry_time = expiry_time
       self.num_args = num_args
       self.lock = threading.Lock() if locked else None

   def __call__(self, func):
       import time
       def wrapped(*args):
           mem_args = args[:self.num_args]
           if mem_args in self.cache:
               result, timestamp = self.cache[mem_args]
               age = time.time() - timestamp
               if not self.expiry_time or age < self.expiry_time:
                   return result
           result = func(*args)
           self.cache[mem_args] = (result, time.time())
           return result
       def locked_wrapped(*args):
           with self.lock:
               return wrapped(*args)
       return wrapped if self.lock is None else locked_wrapped

CREATE_ENGINE_KWARGS = {'pool_recycle': 60}

@memoize()
def get_engine(e):
    from sqlalchemy import create_engine
    return create_engine(e, **CREATE_ENGINE_KWARGS)

def format_sql_list_param(strs):
    strs = [str(s) for s in strs]
    if len(strs) == 0: return "(null)"
    for s in strs:
        assert "'" not in s, "{} has ' in it".format(s)
    return "('{}')".format("', '".join(strs))

def get_resultproxy_as_dict(result, dict=dict):
    def helper():
        keys = result.keys()
        for r in result:
            yield dict((k, v) for k, v in zip(keys, r))
    return list(helper())

def _format_query_list_key(key, query, params):
    values = params.pop(key[1:])
    new_keys = []
    for i, value in enumerate(values):
        new_key = '{}_{}'.format(key, i)
        new_keys.append(new_key)
        params[new_key[1:]] = value
    new_keys_str = ", ".join(new_keys) or "null"
    query = query.replace(key, "({})".format(new_keys_str))
    return query, params

def _format_query_tuple_list_key(key, query, params):
    values = params.pop(key[1:])
    new_keys = []
    for i, value in enumerate(values):
        new_key = '{}_{}'.format(key, i)
        assert isinstance(value, tuple)
        new_keys2 = []
        for i, tuple_val in enumerate(value):
            new_key2 = '{}_{}'.format(new_key, i)
            new_keys2.append(new_key2)
            params[new_key2[1:]] = tuple_val
        new_keys.append("({})".format(", ".join(new_keys2)))
    new_keys_str = ", ".join(new_keys) or "null"
    query = query.replace(key, "({})".format(new_keys_str))
    return query, params

def format_query_with_list_params(query, params):
    import re
    keys = set(re.findall("(?P<key>:[a-zA-Z_]+_list)", query))
    for key in keys:
        if key.endswith('_tuple_list'):
            query, params = _format_query_tuple_list_key(key, query, params)
        else:
            query, params = _format_query_list_key(key, query, params)
    return query, params

def run_threads(threads, target):
    import threading, time
    def start_thread(i):
        t = threading.Thread(target=target, args=[i])
        t.daemon = True
        t.start()
        return t
    [start_thread(i+1) for i in range(threads)]
    while threading.active_count() > 1:
        time.sleep(0.5)


class threadsafe_iter(object):
    def __init__(self, it):
        import threading
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

    next = __next__

def run_parallel_nice(num_threads, fn, iterable, log=True, logger=None, exit=False):
    import signal, logging
    if log and not logger:
        logger = logging.getLogger()
    callback = None if not logger else lambda: logger.info("Graceful interrupt")
    iterable = threadsafe_iter(iter(iterable))
    has_errors = [False]
    with GracefulInterruptHandler(sig=signal.SIGINT, exit=exit, callback=callback) as h:
        def thread(i):
            while True:
                if h.interrupted:
                    if logger: logger.info("Thread {} exiting".format(i))
                    return
                try: key = next(iterable)
                except StopIteration: return
                try: fn(key)
                except Exception:
                    if logger: logger.warn("Error while processing {}".format(key), exc_info=1)
                    has_errors[0] = True
        if num_threads == 1:
            thread(0)
        else:
            run_threads(num_threads, thread)
        return has_errors[0]

class GracefulInterruptHandler(object):
    def __init__(self, sig=None, callback=None, exit=False):
        import signal
        self.sig = sig or signal.SIGINT
        self.callback = callback
        self.exit = exit

    def __enter__(self):
        import signal
        self.interrupted = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)
        def handler(signum, frame):
            self.release()
            self.interrupted = True
            if self.callback: self.callback()
        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        self.release()
        if self.exit:
            import sys
            sys.exit(0)

    def release(self):
        import signal
        if self.released:
            return False
        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True


def execute_sql(engine, query, engines={}, **kwargs):
    from sqlalchemy.sql import text
    if isinstance(engine, basestring):
        engine = get_engine(engines.get(engine, engine))
    is_session = 'session' in repr(engine.__class__).lower()
    query, kwargs = format_query_with_list_params(query, kwargs)

    q = text(query.format(**kwargs))
    return engine.execute(q, params=kwargs) if is_session else engine.execute(q, **kwargs)

def sql(*args, **kwargs):
    return SqlProxy(execute_sql(*args, **kwargs))

class ObjProxy(object):
    def __init__(self, proxied):
        self._proxied = proxied

    def __iter__(self):
        return self._proxied.__iter__()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._proxied, attr)

class SqlProxy(ObjProxy):
    def dicts_iter(self, dict=dict):
        result = self._proxied
        keys = result.keys()
        for r in result:
            yield dict((k, v) for k, v in zip(keys, r))

    def pk_map_iter(self):
        result = self._proxied
        keys = result.keys()
        for r in result:
            yield (r[0], dict((k, v) for k, v in zip(keys, r)))

    def kv_map_iter(self):
        result = self._proxied
        keys = result.keys()
        for r in result:
            yield (r[0], r[1])

    def scalars_iter(self):
        result = self._proxied
        for r in result:
            yield r[0]

    def pk_map(self, dict=dict):
        return dict(self.pk_map_iter())

    def kv_map(self, dict=dict):
        return dict(self.kv_map_iter())

    def dicts(self, dict=dict):
        return list(self.dicts_iter(dict=dict))

    def scalars(self):
        return list(self.scalars_iter())

    def scalar_set(self):
        return set(self.scalars_iter())

    def dict(self, dict=dict):
        try:
            return self.dicts(dict=dict)[0]
        except IndexError:
            return None

def get_results_as_dict(*args, **kwargs):
    return list(get_results_as_dict_iter(*args, **kwargs))

def get_results_as_dict_iter(engine, query, dict=dict, engines={}, **kwargs):
    from sqlalchemy.sql import text
    if isinstance(engine, basestring):
        engine = get_engine(engines.get(engine, engine))
    is_session = 'session' in repr(engine.__class__).lower()
    query, kwargs = format_query_with_list_params(query, kwargs)

    q = text(query.format(**kwargs))
    result = engine.execute(q, params=kwargs) if is_session else engine.execute(q, **kwargs)
    keys = result.keys()
    for r in result:
        yield dict((k, v) for k, v in zip(keys, r))

try:
    import redis

    def redis_pipeline(fn):
        import functools
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if 'pipeline' in kwargs:
                return fn(self, *args, **kwargs)
            with self.pipeline() as p:
                fn(self, pipeline=p, *args, **kwargs)
                return p.execute()
        return wrapper

    class RedisClient(redis.StrictRedis):
        @redis_pipeline
        def replace_hash(self, key, values, pipeline):
            import json, time
            temp_key = 'tmp:{}:{}'.format(int(time.time()), key)
            if hasattr(values, 'items'):
                values = values.items()
            values = list(values)
            if not values:
                pipeline.delete(key)
            else:
                for k, v in values:
                    pipeline.hset(temp_key, k,v if  isinstance(v, basestring) else json.dumps(v))
                pipeline.rename(temp_key, key)

        def hget_multiple(self, hkey, *keys):
            with self.pipeline() as p:
                for key in keys:
                    p.hget(hkey, key)
                return p.execute()

        def hget_all(self, hkey):
            import collections
            keys = self.hgetall(hkey)
            mapping = collections.defaultdict(list)
            for key in keys:
                mapping[key] = keys.get(key)

            return mapping

except ImportError: pass

def json_default(o):
    import datetime, decimal
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if type(o) is decimal.Decimal:
        return float(o)
    if type(o) is bytearray:
        return o.decode('utf8')
    return o

def as_json(support_jsonp=False, sort_keys=False):
    from functools import wraps
    from flask import request, Response
    import json

    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            ret = fn(*args, **kwargs)
            status = 200
            if isinstance(ret, tuple):
                status, ret = ret
            if isinstance(ret, dict) or isinstance(ret, list):
                content = json.dumps(ret, default=json_default, sort_keys=sort_keys)
                callback = request.args.get('callback', False)
                if support_jsonp and callback:
                    content = "{callback}({content})".format(callback=callback, content=content)
                return Response(content, status=status, mimetype='application/json')
            else:
                return ret
        return inner
    return decorator

def as_content_type(content_type):
    from functools import wraps
    from flask import Response

    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            ret = fn(*args, **kwargs)
            return Response(ret, content_type=content_type)
        return inner
    return decorator

as_xml = lambda: as_content_type('text/xml; charset=utf-8')

def support_jsonp(f):
    from functools import wraps
    from flask import request, current_app
    """Wraps JSONified output for JSONP"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        callback = request.args.get('callback', False)
        if callback:
            content = str(callback) + '(' + str(f(*args,**kwargs).data) + ')'
            return current_app.response_class(content, mimetype='application/javascript')
        else:
            return f(*args, **kwargs)
    return decorated_function

def add_response_headers(headers={}):
    from flask import make_response
    from functools import wraps
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            resp = make_response(f(*args, **kwargs))
            h = resp.headers
            for header, value in headers.items():
                h[header] = value
            return resp
        return decorated_function
    return decorator

class attr_accessor(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        cur = self.obj
        for p in key.split('.'):
            try: p = int(p)
            except Exception: pass
            try: cur = cur[p]
            except Exception: cur = getattr(cur, p)
        return cur

def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    import fnmatch
    from datetime import timedelta
    from functools import update_wrapper
    from flask import current_app, request, make_response
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if isinstance(origin, basestring):
        origin = [origin]
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            #if 'Origin' in request.headers and len(fnmatch.filter(origin, request.headers['Origin'])) > 0:
            if 'Origin' in request.headers and any(fnmatch.fnmatch(request.headers['Origin'], o) for o in origin):
                h['Access-Control-Allow-Origin'] = request.headers['Origin']
            else:
                h['Access-Control-Allow-Origin'] = 'null'


            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator



def retry(ExceptionToCheck, tries=4, delay=3, backoff=2, logger=None):
    """Retry calling the decorated function using an exponential backoff.

    http://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
    original from: http://wiki.python.org/moin/PythonDecoratorLibrary#Retry

    :param ExceptionToCheck: the exception to check. may be a tuple of
        exceptions to check
    :type ExceptionToCheck: Exception or tuple
    :param tries: number of times to try (not retry) before giving up
    :type tries: int
    :param delay: initial delay between retries in seconds
    :type delay: int
    :param backoff: backoff multiplier e.g. value of 2 will double the delay
        each retry
    :type backoff: int
    :param logger: logger to use. If None, print
    :type logger: logging.Logger instance
    """
    import time, functools
    def deco_retry(f):

        @functools.wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except ExceptionToCheck as e:
                    msg = "%s, Retrying in %d seconds..." % (str(e), mdelay)
                    if logger:
                        logger.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry

def grouper(n, iterable):
    import itertools
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk: return
       yield chunk


def get_google_credentials(scopes, **kwargs):
    from oauth2client.client import GoogleCredentials
    credentials = GoogleCredentials.get_application_default()
    credentials._kwargs = kwargs
    return credentials.create_scoped(scopes)


def get_google_service(scopes, *args):
    credentials = get_google_credentials(scopes)

    import httplib2; from apiclient.discovery import build; http = httplib2.Http()
    http = credentials.authorize(http)
    return build(*args, http=http, cache_discovery=False)

import contextlib
@contextlib.contextmanager
def chdir(dirname=None, delete_after=False):
    import os
    curdir = os.getcwd()
    try:
        if dirname is not None:  os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)
        if dirname and delete_after:
            import shutil
            shutil.rmtree(dirname)

def md5_for_file(f, block_size=2**20):
    import hashlib
    md5 = hashlib.md5()
    if isinstance(f, basestring):
        f = open(f, 'rb')
    while True:
        data = f.read(block_size)
        if not data:
            break
        md5.update(data)
    return md5.hexdigest()

def groupby(data, keyfn):
    import itertools
    return itertools.groupby(sorted(data, key=keyfn), keyfn)

def decode_base64(data):
    import base64
    data = str(data).strip()
    missing_padding = 4 - len(data) % 4
    if missing_padding:
        data += b'='* missing_padding
    return base64.decodestring(data)

def requires_basicauth(auth_map):
    from functools import wraps
    from flask import request, Response
    def fn(f):
        if len(auth_map) == 0:
            return f
        @wraps(f)
        def decorated(*args, **kwargs):
            auth = request.authorization
            if not auth or auth_map.get(auth.username) != auth.password:
                return Response('Login Required', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
            return f(*args, **kwargs)
        return decorated
    return fn

class JWTAuth(object):
    def __init__(self, keyfile, **authconfig):
        import jwt
        self.verify = authconfig.pop('verify', True)
        self.key = None if keyfile is None else jwt.rsa_load_pub(keyfile)
        assert not (self.verify and self.key is None), 'must specify keyfile'
        self.config = authconfig

    def decode(self, request):
        import jwt
        try:
            data = jwt.decode(self.get_cookie(request), self.key, verify=self.verify)
            self.set_user(request, data)
            return data
        except Exception: return {}

    def redirect(self, request):
        import urllib
        return self.get_redirect(request, "{}?next={}".format(self.config['auth_url'], urllib.quote(self.get_url(request), safe="")))

    def should_redirect(self, request):
        data = self.decode(request)
        if not data:
            return self.redirect(request)

    @staticmethod
    def check_user(user, **kwargs):
        if not kwargs: return True
        def check_kv(key, val):
            if isinstance(val, basestring): val = set([val])
            else: val = set(val)

            userval = user.get(key, [])
            if isinstance(userval, basestring): userval = set([userval])
            else: userval = set(userval)

            return len(val & userval) > 0
        return any(check_kv(k, v) for k, v in kwargs.items())

    def required(self, *check_fns, **check_args):
        check_fns = list(check_fns)
        if check_args:
            check_fns.append(lambda user: self.check_user(user, **check_args))
        def decorator(fn):
            import functools
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                request = self.get_request(args, kwargs)
                user = self.decode(request)
                if not user:
                    return self.redirect(request)
                if not all(check_fn(user) for check_fn in check_fns):
                    return self.unauthorized(request)
                return fn(*args, **kwargs)
            return wrapper
        return decorator

class JWTAuthDjango(JWTAuth):
    def get_request(self, args=[], kwargs={}):
        return args[0]

    def get_cookie(self, request):
        return request.COOKIES.get(self.config['cookie_name'])

    def get_url(self, request): return request.build_absolute_uri()

    def get_redirect(self, request, url):
        from django.shortcuts import redirect
        return redirect(url)

    def unauthorized(self, request):
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("403 Unauthorized")

    def set_user(self, request, data):
        setattr(request, 'jwt_user', data)


class JWTAuthFlask(JWTAuth):
    def before_request(self):
        from flask import request
        self.set_user(request, self.decode(request))

    def get_request(self, args=[], kwargs={}):
        from flask import request
        return request

    def get_cookie(self, request):
        return request.cookies.get(self.config['cookie_name'])

    def get_url(self, request): return request.url

    def get_redirect(self, request, url):
        from flask import redirect
        return redirect(url)

    def unauthorized(self, request):
        from flask import abort
        return abort(403)

    def set_user(self, request, data):
        from flask import g
        setattr(g, 'jwt_user', data)

class JWTFakeAuthDjango(JWTAuthDjango):
    def __init__(self, data):
        self.data = data

    def decode(self, request):
        self.set_user(request, self.data)
        return self.data

class CountingIterator(object):
    def __init__(self, it):
        self.it = it
        self.count = 0
    def __iter__(self):
        return self
    def next(self):
        nxt = next(self.it)
        self.count += 1
        return nxt
    __next__ = next

class EnvironOverride(object):
    def __init__(self, app, env):
        self.app = app
        self.env = {k: v for k, v in env.items() if v}

    def __call__(self, environ, start_response):
        environ.update(self.env)
        return self.app(environ, start_response)

def NamProxyFix(flask_app):
    if 'PROXY_FIX' in flask_app.config:
        flask_app.wsgi_app = EnvironOverride(flask_app.wsgi_app, flask_app.config['PROXY_FIX'])
    return flask_app

class FakeTable(object):
    def __init__(self, name, columns):
        self.name = name
        self._columns = {c: None for c in columns}

def insert_into(engine, table, rows, engines={}, tables={}):
    from sqlalchemy import text
    if not rows: return
    if isinstance(table, basestring): table = tables[table]
    if isinstance(engine, basestring):
        engine = get_engine(engines.get(engine, engine))
    for row in rows:
        columns = list(set(table._columns.keys()) & set(row.keys()))
        query = text("INSERT INTO `{}` ({}) VALUES ({}) ON DUPLICATE KEY UPDATE {}".format(table.name, ", ".join(columns), ", ".join(":" + c for c
in columns), ", ".join("{}=VALUES({})".format(c, c) for c in columns)))
        engine.execute(query, row)
    return engine

def insert_ignore_into(engine, table, rows, engines={}, tables={}):
    from sqlalchemy import text
    if not rows: return
    if isinstance(table, basestring): table = tables[table]
    if isinstance(engine, basestring):
        engine = get_engine(engines.get(engine, engine))
    for row in rows:
        columns = list(set(table._columns.keys()) & set(row.keys()))
        query = text("INSERT IGNORE INTO `{}` ({}) VALUES ({})".format(table.name, ", ".join(columns), ", ".join(":" + c for c in columns)))
        engine.execute(query, row)
    return engine

def insert_into_batch(engine, table, rows, batch=500, engines={}, tables={}):
    from sqlalchemy import text
    try: row = next(rows)
    except StopIteration: return 0
    if isinstance(table, basestring): table = tables[table]
    if isinstance(engine, basestring):
        engine = get_engine(engines.get(engine, engine))
    columns = list(set(table._columns.keys()) & set(row.keys()))
    query = text("INSERT INTO `{}` ({}) VALUES ({}) ON DUPLICATE KEY UPDATE {}".format(table.name, ", ".join(columns), ", ".join(":" + c for c in columns), ", ".join("{}=VALUES({})".format(c, c) for c in columns)))
    engine.execute(query, row)
    for group in grouper(batch, rows):
        engine.execute(query, group)
    return engine

def get_days_ago(days=1):
    import datetime
    return (datetime.datetime.now() - datetime.timedelta(days)).strftime("%Y%m%d")

def date_range(start_date=None, end_date=None):
    import datetime

    start_date = start_date or get_days_ago()
    if '-' in start_date:
        start_date, end_date = start_date.split('-', 1)
    end_date = end_date or start_date

    start_date = (start_date+'01')[:8]
    end_date = (end_date+'01')[:8]

    start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")

    for n in range(int((end_date - start_date).days) + 1):
        yield (start_date + datetime.timedelta(days=n)).strftime("%Y%m%d")

def month_range(*args, **kwargs):
    from collections import OrderedDict
    ret = OrderedDict()
    dates = list(date_range(*args, **kwargs))
    for date in dates:
        ret[date[:6]] = True
    return ret.keys()

class TimeoutException(Exception): pass

@contextlib.contextmanager
def signal_timeout(seconds, exception=TimeoutException):
    # http://stackoverflow.com/a/601168
    import signal
    def signal_handler(signum, frame):
        raise exception()
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class ThreadedWorkers(object):
    def __init__(self, logger=None):
        self.workers = []
        self.logger = logger or get_logger('workers')

    def register(self, fn):
        self.workers.append(fn)
        return fn

    def launch(self, target):
        import threading
        t = threading.Thread(target=target)
        t.daemon = True
        t.start()
        self.logger.info('started worker {}'.format(target.__name__))
        return t

    def main(self):
        import time
        workers = {fn: self.launch(fn) for fn in self.workers}
        while True:
            workers = {fn: (thread if thread.isAlive() else self.launch(fn)) for fn, thread in workers.items()}
            time.sleep(30)

