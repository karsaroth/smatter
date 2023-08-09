from __future__ import annotations
import io
import os
from pathlib import Path
from typing import Optional, Set
import uuid
import loguru
import json
import av
import threading
import multiprocessing as mp
from .utils import QueueIO
from multiprocessing.synchronize import Event
from aiohttp import web
import aiohttp.abc as web_abc
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel, RTCConfiguration
from aiortc.contrib.media import MediaPlayer, PlayerStreamTrack, REAL_TIME_FORMATS  

class SmatterRTCServer():
  def __init__(self, stop: Event, _logger: loguru.Logger, passthrough_queue: mp.Queue, subtitle_queue: mp.Queue, file_root=Path('./')):
    self.stopper = stop
    self.file_root = file_root
    self.peer_connections = set()
    self._logger = _logger
    self.passthrough_queue = passthrough_queue
    self.subtitle_queue = subtitle_queue
    self.pipe = QueueIO('r', passthrough_queue)
    self.media_player = MediaPlayer(file=self.pipe)
    self.end_signal = mp.Event()

  def prep_shutdown(self):
    async def on_shutdown(app):
      # close peer connections
      _results = [await pc.close() for pc in self.peer_connections]
      self.peer_connections.clear()
    return on_shutdown
  
  def prep_index(self):
    async def index(request: web.Request):
      content = open(os.path.join(self.file_root, "index.html"), "r").read()
      return web.Response(content_type="text/html", text=content)
    return index

  def prep_offer(self):
    async def offer(request: web.Request):
      params = await request.json()

      pc = RTCPeerConnection()

      pc_id = "PeerConnection(%s)" % uuid.uuid4()
      self.peer_connections.add(pc)

      def log_info(msg, *args):
        self._logger.info(pc_id + " " + msg, *args)

      log_info(f"Created for {request.remote}")

      @pc.on("connectionstatechange")
      async def on_connectionstatechange():
          log_info(f"Connection state is {pc.connectionState}")
          if pc.connectionState == "failed":
              await pc.close()
              self.peer_connections.discard(pc)

      pc.addTrack(self.media_player.audio)
      pc.addTrack(self.media_player.video)
      data_channel = pc.createDataChannel(
        'smatter_vtt'
      )
      
      def send_subs():
        log_info('Started Sending Subs to Client')
        if self.end_signal.is_set():
          return
        next = (0, 0, 'DUMMY')
        while not self.stopper.is_set() and \
            data_channel.bufferedAmount <= data_channel.bufferedAmountLowThreshold and \
            (next:= self.subtitle_queue.get()):
          s, e, t = next
          msg = {
            'start': round(s, 2),
            'end': round(e, 2),
            'text': t if t else ''
          }
          log_info(f'Sending subtitle to client: {json.dumps(msg)}')
          data_channel.send(json.dumps(msg))
        if not next:
          self.end_signal.set()

      data_channel.on('bufferedamountlow', send_subs)
      data_channel.on('open', send_subs)

      # handle offer
      offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
      await pc.setRemoteDescription(offer)

      # send answer
      answer = await pc.createAnswer()
      if answer is not None:
        await pc.setLocalDescription(answer)

      return web.Response(
          content_type="application/json",
          text=json.dumps(
              {
                "sdp": pc.localDescription.sdp, 
                "type": pc.localDescription.type
              }
          ),
      )
    return offer
  
class WebLogger(web_abc.AbstractAccessLogger):
  def __init__(self, logger: loguru.Logger):
    self._logger = logger

  async def log(self, request: web.BaseRequest, response: web.StreamResponse, time: float):
    self._logger.info('Remote: {remote}, Method: {method}, Path: {path}, Status: {status}',
      remote=request.remote, method=request.method, path=request.path, status=response.status)
    self._logger.debug('Response Body Length: {body_len}', response.body_length)


def web_rtc_server(
    stop: Event,
    _logger: loguru.Logger,
    passthrough_queue: mp.Queue,
    subtitle_queue: mp.Queue,
    front_end_dir: Path,
    host: str,
    port: int
  ):
  app = web.Application()
  rtcs = SmatterRTCServer(stop, _logger, passthrough_queue, subtitle_queue, front_end_dir)
  app.on_shutdown.append(rtcs.prep_shutdown())
  app.router.add_get("/", rtcs.prep_index())
  app.router.add_get("/index.html", rtcs.prep_index())
  app.router.add_routes([
    web.static('/js', Path(front_end_dir) / 'js', append_version=True),
    web.static('/css', Path(front_end_dir) / 'css', append_version=True),
  ])
  app.router.add_post("/offer", rtcs.prep_offer())
  web.run_app(
      app, host=host, port=port
  )
  return app