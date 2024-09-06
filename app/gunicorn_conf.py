"""Gunicorn configuration file."""
import json
import os

workers_per_core_str = os.getenv("WORKERS_PER_CORE", "4")
max_workers_str = os.getenv("MAX_WORKERS")
use_max_workers = None # pylint: disable=invalid-name
if max_workers_str:
    use_max_workers = int(max_workers_str)
web_concurrency_str = os.getenv("WEB_CONCURRENCY", None)

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8080")
root_path = os.getenv("ROOT_PATH", "/")
bind_env = os.getenv("BIND", None)
use_loglevel = os.getenv("LOG_LEVEL", "info")
if bind_env:
    use_bind = bind_env
else:
    use_bind = f"{host}:{port}"


accesslog_var = os.getenv("ACCESS_LOG", "-")
use_accesslog = accesslog_var or None
errorlog_var = os.getenv("ERROR_LOG", "-")
use_errorlog = errorlog_var or None
graceful_timeout_str = os.getenv("GRACEFUL_TIMEOUT", "1200")
timeout_str = os.getenv("TIMEOUT", "1200")
keepalive_str = os.getenv("KEEP_ALIVE", "5")

# Gunicorn config variables
loglevel = use_loglevel
# workers =1
bind = use_bind
errorlog = use_errorlog
# worker_tmp_dir = "/dev/shm"
accesslog = use_accesslog
graceful_timeout = int(graceful_timeout_str)
timeout = int(timeout_str)
keepalive = int(keepalive_str)


if os.getenv("CERT_PATH", None) is not None:
    certfile = os.getenv("CERT_PATH", None)
if os.getenv("KEY_PATH", None) is not None:
    keyfile = os.getenv("KEY_PATH", None)



log_data = {
    "loglevel": loglevel,
    # "workers": workers,
    "bind": bind,
    "graceful_timeout": graceful_timeout,
    "timeout": timeout,
    "keepalive": keepalive,
    "errorlog": errorlog,
    "accesslog": accesslog,
    "use_max_workers": use_max_workers,
    "host": host,
    "port": port,
}
print(json.dumps(log_data))
