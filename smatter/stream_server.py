from __future__ import annotations
import os
import re
import queue
import json
import multiprocessing as mp
import threading as th
import asyncio
from pathlib import Path
from typing import Dict, Literal, Optional, TypedDict, List, Any
import loguru
from aiohttp import web
from smatter.transx import InteractiveTransXProcess
import smatter.utils as u
import smatter.ff_process as ff

class MinimalTransXConfig(TypedDict):
  """
  Only the data that
  shouldn't change in
  streaming mode, the
  rest is dynamic
  """
  _logger: loguru.Logger
  base_path: str
  model_config: TypedDict

class StreamState(TypedDict):
  """
  Keep track of stateful data
  for a Smatter stream
  """
  transx: InteractiveTransXProcess
  stream_input: ff.ProcessWrappedStreamInput
  stream_output: ff.MultiprocessStreamOutput
  threads: List[th.Thread]
  logs: List[str]
  cache_dir: str
  stream_url: Optional[str]
  requested_start: Optional[str]
  language: Optional[str]
  goal: Optional[Literal['translate', 'transcribe']]
  quality: Optional[str]


class StreamRequestHandler:
  """
  A handler class for request
  related to watching a stream
  via smatter
  """
  def __init__(self,
               _logger: loguru.Logger,
               www: Path,
               stream: Path,
               minimal_config: MinimalTransXConfig,
      ):
    self.www = www
    self.stream = stream
    self.output_queue_complete = False
    self.status = {
      'model_loading': mp.Event(),
      'model_loaded': mp.Event()
    }
    self.state = StreamState(
      transx = InteractiveTransXProcess(
        minimal_config['_logger'],
        minimal_config['model_config']
      ),
      stream_input = ff.ProcessWrappedStreamInput(
        minimal_config['_logger'],
        ff.StreamInputConfig(
          url='',
          start='',
          cache_dir=minimal_config['base_path'],
          quality=''
        )
      ),
      stream_output= ff.MultiprocessStreamOutput(
        minimal_config['_logger'],
        base_dir=stream,
        length=6
      ),
      threads = [],
      logs = [],
      cache_dir = minimal_config['base_path'],
      stream_url=None,
      requested_start=None,
      language=None,
      goal=None,
      quality=None,
    )
    self.logger = _logger
    self.subtitle_buffer = []

  def get_www_content(self, file: str, r_type: str, c_type: str):
    """
    Dynamic handling of
    content in the www
    directory
    """
    async def handler(_request: web.Request):
      content = open(os.path.join(self.www, file), r_type).read()
      return web.Response(content_type=c_type, body=content)
    return handler

  def get_stream_content(self):
    """
    Getting stream content
    with appropriate
    content type
    """
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
          return web.Response(status=web.HTTPServiceUnavailable.status_code)
        content = open(os.path.join(self.stream, file), r_type).read()
        return web.Response(content_type=c_type, body=content)
      return web.Response(status=web.HTTPNotFound.status_code)
    return handler

  def get_subtitles(self):
    """
    Handle getting currently
    available subtitles
    """
    async def handler(_request: web.Request):
      response = self.subtitle_buffer.copy()
      while not self.output_queue_complete and \
            self.state['transx'] and \
            not self.state['transx'].output_queue.empty():
        try:
          queue_item = self.state['transx'].output_queue.get_nowait()
          if queue_item is None:
            self.output_queue_complete = True
            break
          start, end, text = queue_item
          sub_record = {
            'start': start,
            'end': end,
            'text': text
          }
          response.append(sub_record)
          self.subtitle_buffer.append(sub_record)
        except queue.Empty:
          break
        except ValueError:
          break
        except Exception as ex:
          self.state['logs'].append(str(ex))
          self.logger.exception(ex)
          return web.Response(status=web.HTTPInternalServerError.status_code)
      return web.json_response(response)
    return handler

  def __ready_for_stream(self):
    """
    Check if the stream
    is ready to be started
    """
    return all([
      self.status['model_loaded'].is_set(),
      self.state['stream_url'],
      self.state['language'],
      self.state['goal'],
      self.state['quality']
    ])

  def __handle_thread(self, thread: th.Thread):
    """
    A thread that is used
    to run the stream
    """
    self.state['threads'] = [t for t in self.state['threads'] if t.is_alive()]
    thread.start()
    self.state['threads'].append(thread)

  def __start(self):
    """
    Start the stream and
    transx processes
    """
    self.state['transx'].input_queue = self.state['stream_input'].pcm_queue
    self.state['stream_output'].passthrough_queue = \
        self.state['stream_input'].passthrough_queue
    self.state['stream_input'].input_config = ff.StreamInputConfig(
      url=self.state['stream_url'], # type: ignore
      start=self.state['requested_start'] if self.state['requested_start'] else '0',
      cache_dir=self.state['cache_dir'],
      quality=self.state['quality'] # type: ignore
    )
    self.state['stream_input'].start()
    self.state['transx'].start(
      self.state['requested_start'] if self.state['requested_start'] else '0',
      self.state['language'],
      self.state['goal']
    )
    self.state['stream_output'].start()

  def __load(self, request_data):
    """
    Inital loading of
    transx model
    """
    self.status['model_loading'].set()
    def check_model():
      try:
        self.state['transx'].check_model(
          request_data['download'] if 'download' in request_data else True
        )
        self.status['model_loading'].clear()
        self.status['model_loaded'].set()
      except Exception as ex:
        self.state['logs'].append(str(ex))
        self.logger.exception(ex)
        self.status['model_loading'].clear()
        self.status['model_loaded'].clear()
    self.__handle_thread(th.Thread(target=check_model))

  def __set_state(self, request_data):
    """
    Configure the stream
    settings
    """
    if request_data['requested_start']:
      if u.hms_match(request_data['requested_start']):
        self.state['requested_start'] = request_data['requested_start']
    self.state['stream_url'] = request_data['stream_url'] \
      if 'stream_url' in request_data else None
    self.state['language'] = request_data['language'] \
      if 'language' in request_data else None
    self.state['goal'] = request_data['goal'] \
      if 'goal' in request_data else None
    self.state['quality'] = request_data['quality'] \
      if 'quality' in request_data else None

  def get_status(self):
    """
    Handler to give user
    status info about the
    back end processes
    """
    async def handler(request: web.Request):
      return web.json_response({
        'model': self.status['model_loaded'].is_set(),
        'stream': self.__ready_for_stream(),
        'stream_input': {
          'running': self.state['stream_input'].is_running(),
          'detail': self.state['stream_input'].status_detail()
        },
        'stream_output': {
          'running': self.state['stream_output'].is_running(),
          'detail': self.state['stream_output'].status_detail()
        },
        'transx': {
          'running': self.state['transx'].status(),
          'detail': self.state['transx'].status_detail()
        },
        'logs': self.state['logs']
      })

    return handler

  def post_state_change(self):
    """
    Handler for user requested
    state changes, loading models
    switching streams, etc.
    """
    async def handler(request: web.Request):
      status = web.HTTPAccepted.status_code
      if request.content_type != 'application/json':
        status = web.HTTPUnsupportedMediaType.status_code
      try:
        request_data = await request.json()
        match request_data['action']:
          case 'load':
            if self.status['model_loading'].is_set() \
                or self.status['model_loaded'].is_set() \
                or self.output_queue_complete \
                or self.state['transx'].status():
              return web.StreamResponse(status=web.HTTPConflict.status_code)
            self.__load(request_data)
          case 'set':
            self.__set_state(request_data)
          case 'start':
            if not self.__ready_for_stream():
              return web.StreamResponse(status=web.HTTPConflict.status_code)
            self.__start()
          case 'stop':
            self.state['stream_input'].stop()
            self.state['transx'].stop()
            self.state['stream_output'].stop()
            self.__handle_thread(
              th.Thread(
                name='transx_cleanup',
                target=self.state['transx'].cleanup
              )
            )
            self.__handle_thread(
              th.Thread(
                name='stream_output_cleanup',
                target=self.state['stream_output'].cleanup
              )
            )
            self.subtitle_buffer = []
          case _:
            status=web.HTTPBadRequest.status_code
      except json.JSONDecodeError:
        status=web.HTTPBadRequest.status_code
      except Exception as ex:
        self.state['logs'].append(str(ex))
        self.logger.exception(ex)
        status=web.HTTPInternalServerError.status_code
      return web.StreamResponse(status=status)

    return handler

def run_server(
    _logger: loguru.Logger,
    host: str,
    port: int,
    www: Path,
    stream: Path,
    transx_config: MinimalTransXConfig,
  ):
  """
  Runs a Smatter streaming
  HTTP server
  """
  def run_server_thread():
    site: web.TCPSite | None = None
    runner: web.AppRunner | None = None
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = web.Application()
    try:
      srh = StreamRequestHandler(_logger, www, stream, transx_config)
      app.router.add_get('/', srh.get_www_content('index.html', 'r', 'text/html'))
      app.router.add_get(
        '/js/smatter.js', 
        srh.get_www_content('js/smatter.js', 'r', 'text/javascript')
      )
      app.router.add_get('/favicon.ico', srh.get_www_content('favicon.ico', 'rb', 'image/x-icon'))
      app.router.add_get('/index.html', srh.get_www_content('index.html', 'r', 'text/html'))
      app.router.add_get('/poster.png', srh.get_www_content('poster.png', 'rb', 'image/png'))
      app.router.add_get("/stream/{file}", srh.get_stream_content())
      app.router.add_get("/smatter/subtitles", srh.get_subtitles())
      app.router.add_post("/smatter/state", srh.post_state_change())
      app.router.add_get("/smatter/status", srh.get_status())
      runner = web.AppRunner(
        app,
        handle_signals=True,
      )
      loop.run_until_complete(runner.setup())
      site = web.TCPSite(runner, host, port)
      loop.run_until_complete(site.start())
      loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
      _logger.info('Stopping stream server')
      if site:
        loop.run_until_complete(site.stop())
      if runner:
        loop.run_until_complete(runner.cleanup())
    except Exception as ex:
      _logger.exception(ex)
    finally:
      _logger.info('Closing stream server event loop')
      loop.stop()
      loop.close()
      _logger.info('Server thread stopping')

  thread = th.Thread(target=run_server_thread, name='stream_server', daemon=True)
  thread.start()
  return thread
