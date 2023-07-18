from __future__ import annotations
import os
from pathlib import Path
import uuid
import loguru
import json
import time
import multiprocessing as mp
from .utils import QueueIO
from multiprocessing.synchronize import Event
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

class SmatterRTCServer():
  def __init__(self, stop: Event, _logger: loguru.Logger, passthrough_queue: mp.Queue, file_root=Path('./')):
    self.file_root = file_root
    self.peer_connections = set()
    self._logger = _logger
    self.passthrough_queue = passthrough_queue
    self.pipe = QueueIO('r', passthrough_queue)
    self.media_player = MediaPlayer(file=self.pipe)

  def prep_shutdown(self):
    async def on_shutdown(app):
      # close peer connections
      _results = [await pc.close() for pc in self.peer_connections]
      self.peer_connections.clear()
    return on_shutdown
  
  def prep_index(self):
    async def index(request):
      content = open(os.path.join(self.file_root, "index.html"), "r").read()
      return web.Response(content_type="text/html", text=content)
    return index

  def prep_javascript(self):
    async def javascript(request):
      content = open(os.path.join(self.file_root, "client.js"), "r").read()
      return web.Response(content_type="application/javascript", text=content)
    return javascript

  def prep_offer(self):
    async def offer(request):
      params = await request.json()
      offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

      pc = RTCPeerConnection()
      pc_id = "PeerConnection(%s)" % uuid.uuid4()
      self.peer_connections.add(pc)

      def log_info(msg, *args):
        self._logger.info(pc_id + " " + msg, *args)

      log_info("Created for %s", request.remote)

      @pc.on("connectionstatechange")
      async def on_connectionstatechange():
          log_info("Connection state is %s", pc.connectionState)
          if pc.connectionState == "failed":
              await pc.close()
              self.peer_connections.discard(pc)

      pc.addTrack(self.media_player.audio)
      pc.addTrack(self.media_player.video)

      # handle offer
      await pc.setRemoteDescription(offer)

      # send answer
      answer = await pc.createAnswer()
      if answer is not None:
        await pc.setLocalDescription(answer)

      return web.Response(
          content_type="application/json",
          text=json.dumps(
              {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
          ),
      )
    return offer

def web_rtc_server(
    stop: Event,
    _logger: loguru.Logger,
    passthrough_queue: mp.Queue,
    front_end_dir: Path,
    host: str,
    port: int
  ):
  app = web.Application()
  rtcs = SmatterRTCServer(stop, _logger, passthrough_queue, front_end_dir)
  app.on_shutdown.append(rtcs.prep_shutdown())
  app.router.add_get("/", rtcs.prep_index())
  app.router.add_get("/client.js", rtcs.prep_javascript())
  app.router.add_post("/offer", rtcs.prep_offer())
  web.run_app(
      app, access_log=None, host=host, port=port
  )
  return app