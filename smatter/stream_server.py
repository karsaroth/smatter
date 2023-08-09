from __future__ import annotations
from aiohttp import web
from pathlib import Path
import os, re
import multiprocessing as mp
import aiohttp.abc as web_abc
import loguru

class StreamRequestHandler:
  def __init__(self, www: Path, stream: Path):
    self.www = www
    self.stream = stream

  def get_www_content(self, file: str, r_type: str, c_type: str):
    async def handler(request: web.Request):
      content = open(os.path.join(self.www, file), r_type).read()
      return web.Response(content_type=c_type, body=content)
    return handler
  
  def get_stream_content(self):
    async def handler(request: web.Request):
      match = re.fullmatch(r'/stream/([^/]*\.(m3u8|ts))', request.path)
      if match:
        file = match.group(1)
        ext = match.group(2)
        if ext == 'm3u8':
          c_type = 'application/vnd.apple.mpegurl'
          r_type = 'r'
        elif ext == 'ts':
          c_type = 'video/mp2t'
          r_type = 'rb'
        else:
          return web.Response(status=web.HTTPNotFound.status_code)
        if not os.path.exists(os.path.join(self.stream, file)):
          return web.Response(status=web.HTTPInternalServerError.status_code)
        content = open(os.path.join(self.stream, file), r_type).read()
        return web.Response(content_type=c_type, body=content)
      else:
        return web.Response(status=web.HTTPNotFound.status_code)
    return handler  

def run_server(_logger: loguru.Logger, host: str, port: int, www: Path, stream: Path, subtitles: mp.Queue):
  app = web.Application()
  srh = StreamRequestHandler(www, stream)
  app.router.add_get('/', srh.get_www_content('index.html', 'r', 'text/html'))
  app.router.add_get('/favicon.ico', srh.get_www_content('favicon.ico', 'rb', 'image/x-icon'))
  app.router.add_get('/index.html', srh.get_www_content('index.html', 'r', 'text/html'))
  app.router.add_get('/poster.png', srh.get_www_content('poster.png', 'rb', 'image/png'))
  app.router.add_get("/stream/{file}", srh.get_stream_content())
  # app.router.add_post("/smatter/subtitles", rtcs.prep_offer())
  web.run_app(
      app, host=host, port=port
  )
  return app