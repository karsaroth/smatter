"""Socket server for Smatter."""

from __future__ import annotations

import multiprocessing as mp
import queue
import json
from socketserver import TCPServer, BaseRequestHandler
import socket
import threading
from typing import Any, Callable, Dict, Optional, Tuple
import loguru
from loguru import logger


def run_proccess_server(
  input_queue: mp.Queue,
  output_queue: mp.Queue,
  host: str,
  port: int,
  _logger: loguru.Logger
) -> None:
  """Run socket server backed by multiprocessing queues"""
  run_server(
    input_queue,
    output_queue,
    host,
    port,
    _logger,
    None,
    None
  )

def run_threaded_server(
  input_queue: queue.Queue,
  output_queue: queue.Queue,
  host: str,
  port: int,
  _logger: loguru.Logger,
  con_start_fun: Callable[[], Dict[str, Any]],
  con_stop_fun: Callable[[Dict[str, Any]], None],
) -> None:
  """Run socket server backed by threaded queues"""
  run_server(
    input_queue,
    output_queue,
    host,
    port,
    _logger,
    con_start_fun,
    con_stop_fun,
  )

def run_server(
    input_queue,
    output_queue,
    host: str,
    port: int,
    _logger: loguru.Logger,
    con_start_fun: Optional[Callable[[], Dict[str, Any]]],
    con_stop_fun: Optional[Callable[[Dict[str, Any]], None]]
) -> None:
  """Run socket server."""
  conf = SmatterRequestHandlerConfig(
    _logger,
    forward_to_queue(input_queue),
    read_from_queue(output_queue),
    con_start_fun,
    con_stop_fun
  )

  with TCPServer((host, port), conf.get_handler_fun()) as server:
    logger.info("Listening on {host}:{port}", host=host, port=port)
    server.serve_forever()

def forward_to_queue(input_queue) -> Callable[[bytes], None]:
  """Return function to forward data to an mp/threaded queue."""
  def _forward_to_queue(data: bytes) -> None:
    input_queue.put(data)
  return _forward_to_queue

def read_from_queue(output_queue) -> Callable[[threading.Event], Optional[bytes]]:
  """Return function to read data from an mp/threaded queue."""
  def _read_from_queue(stop: threading.Event) -> Optional[bytes]:
    result: Optional[bytes] = None
    result_recieved = False
    while (not stop.is_set()) and (not result_recieved):
      try:
        result = output_queue.get(True, 0.1)
        result_recieved = True
      except queue.Empty:
        continue
    return result
  return _read_from_queue

class SmatterRequestHandlerConfig():
  """Configuration for SmatterRequestHandler."""
  def __init__(
    self,
    _logger: loguru.Logger,
    input_fun: Callable[[bytes], None],
    output_fun: Callable[[threading.Event], Optional[bytes]],
    con_start_fun: Optional[Callable[[], Dict[str, Any]]],
    con_stop_fun: Optional[Callable[[Dict[str, Any]], None]],
    buffer_size: int = 1024
  ):
    self.logger = _logger
    self.input_fun = input_fun
    self.output_fun = output_fun
    self.con_start_fun = con_start_fun
    self.con_stop_fun = con_stop_fun
    self.buffer_size = buffer_size

  def get_handler_fun(self) -> Callable[[Any, Any, TCPServer], BaseRequestHandler]:
    """Return function to create configured handler."""
    def _configured_handler(request, client_address, server) -> SmatterRequestHandler:
      return SmatterRequestHandler(
        self.logger,
        self.input_fun,
        self.output_fun,
        self.con_start_fun,
        self.con_stop_fun,
        self.buffer_size,
        request,
        client_address,
        server
      )
    return _configured_handler

class SmatterRequestHandler(BaseRequestHandler):
  """Handle audio data stream from client."""

  def __init__(self,
    _logger: loguru.Logger,
    input_fun: Callable[[bytes], None],
    output_fun: Callable[[threading.Event], Optional[bytes]],
    con_start_fun: Optional[Callable[[], Dict[str, Any]]],
    con_stop_fun: Optional[Callable[[Dict[str, Any]], None]],
    buffer_size: int,
    request,
    client_address,
    server
  ):
    self.logger = _logger
    self.input_fun = input_fun
    self.output_fun = output_fun
    self.con_start_fun = con_start_fun
    self.con_stop_fun = con_stop_fun
    self.buffer_size = buffer_size
    super().__init__(request, client_address, server)

  def setup(self) -> None:
    """Setup connection."""
    super().setup()
    if self.con_start_fun is not None:
      self.state = self.con_start_fun()
    if self.logger is None:
      self.logger = loguru.logger
    self.logger.info("Connection from {ca}.", ca=self.client_address)

  def finish(self) -> None:
    """Finish connection."""
    super().finish()
    if self.con_stop_fun is not None:
      self.con_stop_fun(self.state)
    self.logger.info("Connection from {ca} closed.", ca=self.client_address)

  def __read_buffered(self, _socket: socket.socket):
    data = b''
    while True:
      while len(data) < self.buffer_size:
        chunk = _socket.recv(self.buffer_size - len(data))
        if not chunk:
          self.logger.info(
            "Socket connection from {ca} returned no data.",
            ca=self.client_address
          )
          break
        data += chunk
      if len(data) == self.buffer_size:
        yield data
      elif len(data) > self.buffer_size:
        yield data[:self.buffer_size]
        data = data[self.buffer_size:]
      else:
        yield b''
        break
      data = b''

  @staticmethod
  def __write_result(
    _logger: loguru.Logger,
    _socket: socket.socket,
    client_address,
    closed: threading.Event,
    output_fun: Callable[[threading.Event], Optional[bytes]]
  ):
    try:
      while result := output_fun(closed):
        if closed.is_set() or result is None:
          break
        start, end, tx = result
        _socket.sendall(json.dumps({
          "start": start,
          "end": end,
          "tx": tx
        }).encode())
    except ConnectionAbortedError:
      pass
    except OSError:
      pass
    except Exception:
      _logger.exception(
        "Unexpected exception writing to connection from {ca}.",
        ca=client_address
      )
    _logger.info(
      "Write thread is closing for {ca} as the socket is closed.",
      ca=client_address
    )

  def handle(self) -> None:
    """Handle incoming data. Essentially incoming data is buffered and then
    put straight into the input queue. The output queue is read from a separate
    thread and the results are sent back to the client."""
    self.close_event = threading.Event()
    self.thread = threading.Thread(
      target=SmatterRequestHandler.__write_result,
      args=(
        self.logger,
        self.request,
        self.client_address,
        self.close_event,
        self.output_fun
      ),
      name=f"smatter_socker_write_{self.client_address}",
      daemon=True
    )
    self.thread.start()
    try:
      for data in self.__read_buffered(self.request):
        if data:
          self.input_fun(data)
    except ConnectionAbortedError:
      self.logger.info(
        "Socket connection from {ca} aborted.",
        ca=self.client_address
      )
    except Exception:
      self.logger.exception(
        "Error handling connection from {ca}",
        ca=self.client_address
      )
    logger.info("Closing connection from {ca}.", ca=self.client_address)
    self.close_event.set()
    self.thread.join()

@logger.catch
def test_smatter_socket(host: str, port: int, _logger: loguru.Logger) -> None:
  """Test socket server."""

  s = socket.socket()
  s.settimeout(None)
  s.connect((host, port))

  with open("test.wav", "rb") as f:
    while data := f.read(1024):
      s.sendall(data)
      try:
        s.settimeout(0.0)
        output = s.recv(1024)
        if output:
          _logger.info('Server returned {output}', output=output)
      except BlockingIOError:
        continue
      finally:
        s.settimeout(None)
  try:
    s.settimeout(10.0)
    while output := s.recv(1024):
      _logger.info('Server returned {output}', output=output)
  except TimeoutError:
    _logger.info("Server timed out, no more data to receive.")
  s.close()

if __name__ == "__main__":
  test_smatter_socket("localhost", 9999, logger)
