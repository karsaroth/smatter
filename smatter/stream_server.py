from __future__ import annotations
from aiohttp import web
from aiohttp import log
from pathlib import Path
import os
import re
import loguru
import queue
import multiprocessing as mp
import threading as th
import asyncio

class StreamRequestHandler:
  def __init__(self, _logger: loguru.Logger, www: Path, stream: Path, subtitles: mp.Queue):
    self.www = www
    self.stream = stream
    self.subtitle_queue = subtitles
    self.subtitle_queue_complete = False
    self.logger = _logger
    self.subtitle_buffer = []

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
  
  def get_subtitles(self):
    async def handler(request: web.Request):
      response = self.subtitle_buffer.copy()
      while not self.subtitle_queue_complete and not self.subtitle_queue.empty():
        try:
          queue_item = self.subtitle_queue.get_nowait()
          if queue_item == None:
            self.subtitle_queue_complete = True
            break
          s, e, t = queue_item
          sub_record = {
            'start': s,
            'end': e,
            'text': t
          }
          response.append(sub_record)
          self.subtitle_buffer.append(sub_record)
        except queue.Empty:
          break
        except ValueError:
          break
        except Exception as e:
          self.logger.exception(e)
          return web.Response(status=web.HTTPInternalServerError.status_code)
      return web.json_response(response)
    return handler

def run_server(_logger: loguru.Logger, host: str, port: int, www: Path, stream: Path, subtitles: mp.Queue):
  def run_server_thread():
    site: web.TCPSite | None = None
    runner: web.AppRunner | None = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    try:
      srh = StreamRequestHandler(_logger, www, stream, subtitles)
      app.router.add_get('/', srh.get_www_content('index.html', 'r', 'text/html'))
      app.router.add_get('/js/smatter.js', srh.get_www_content('js/smatter.js', 'r', 'text/javascript'))
      app.router.add_get('/favicon.ico', srh.get_www_content('favicon.ico', 'rb', 'image/x-icon'))
      app.router.add_get('/index.html', srh.get_www_content('index.html', 'r', 'text/html'))
      app.router.add_get('/poster.png', srh.get_www_content('poster.png', 'rb', 'image/png'))
      app.router.add_get("/stream/{file}", srh.get_stream_content())
      app.router.add_get("/smatter/subtitles", srh.get_subtitles())
      runner = web.AppRunner(
        app, 
        handle_signals=True,
      )
      loop.run_until_complete(runner.setup())
      site = web.TCPSite(runner, host, port)
      loop.run_until_complete(site.start())
      loop.run_forever()
    except KeyboardInterrupt or SystemExit:
      _logger.info('Stopping stream server')
      if site:
        loop.run_until_complete(site.stop())
      if runner:
        loop.run_until_complete(runner.cleanup())
    except Exception as e:
      _logger.exception(e)
    finally:
      _logger.info('Closing stream server event loop')
      loop.stop()
      loop.close()
      _logger.info('Server thread stopping')

  thread = th.Thread(target=run_server_thread, name='stream_server', daemon=True)
  thread.start()
  return thread