#!/bin/bash

DEFAULT_MODULE_NAME=app.application
MODULE_NAME=${MODULE_NAME:-$DEFAULT_MODULE_NAME}
VARIABLE_NAME=${VARIABLE_NAME:-app}

DEFAULT_PORT=8080
API_PORT=${API_PORT:-$DEFAULT_PORT}

DEFAULT_API_HOST=0.0.0.0
API_HOST=${API_HOST:-$DEFAULT_API_HOST}
echo $API_HOST
APP_MODULE=${APP_MODULE:-"$MODULE_NAME:$VARIABLE_NAME"}

if [[ ! -z "${CERT_PATH}" ]] && [[ ! -z "${KEY_PATH}" ]]; then
echo "using certificates ..."
exec uvicorn $APP_MODULE --host $API_HOST --port $API_PORT --proxy-headers --ssl-certfile "$CERT_PATH" --ssl-keyfile "$KEY_PATH"
else
echo "not using certificates ..."
exec uvicorn $APP_MODULE --host $API_HOST --port $API_PORT --proxy-headers
fi
