"""HTTPS-only upload transport. Credentials never appear in diagnostics."""
import json
import os
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from .api_contracts import encode, MAX_BATCH_BYTES


class UploadError(Exception):
    def __init__(self, code, retryable=False, retry_after=0):
        super().__init__(code)
        self.code, self.retryable, self.retry_after = code,retryable,retry_after


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def configuration():
    url = os.environ.get('RND_MONITOR_API_URL','').rstrip('/')
    key_file = os.environ.get('RND_MONITOR_UPLOAD_KEY_FILE','')
    if not url or not key_file:
        raise UploadError('configuration_missing')
    parts = urlsplit(url)
    if parts.scheme != 'https' or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise UploadError('https_endpoint_required')
    path = Path(key_file).expanduser()
    if os.name != 'nt' and path.stat().st_mode & 0o077:
        raise UploadError('credential_file_must_be_owner_only')
    token = path.read_text(encoding='utf-8').strip()
    if not 40 <= len(token) <= 200 or any(c.isspace() for c in token):
        raise UploadError('invalid_credential_file')
    return url,token


def post(payload, route='batches'):
    url, token = configuration()
    body = encode(payload)
    if len(body)>MAX_BATCH_BYTES:
        raise UploadError('batch_too_large')
    request = Request(url+'/v1/'+route, data=body, headers={
        'Content-Type':'application/json', 'X-RND-Upload-Key':token}, method='POST')
    try:
        with build_opener(NoRedirect()).open(request,timeout=10) as response:
            data = response.read(65537)
            if len(data)>65536:
                raise UploadError('response_too_large',True)
            return json.loads(data)
    except HTTPError as error:
        try:
            retry_after = min(300,max(0,int(error.headers.get('Retry-After','0'))))
        except ValueError:
            retry_after = 0
        raise UploadError(f'http_{error.code}',error.code==429 or error.code>=500,retry_after) from None
    except (URLError,TimeoutError,OSError,ValueError):
        raise UploadError('network_or_response_error',True) from None
