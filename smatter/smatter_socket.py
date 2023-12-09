"""Socket server for Smatter."""

from __future__ import annotations

import multiprocessing as mp
import queue
from socketserver import TCPServer, BaseRequestHandler
import socket
import threading
from typing import Optional

import loguru
from loguru import logger


def run_server(
  input_queue: mp.Queue,
  output_queue: mp.Queue,
  host: str,
  port: int,
  _logger: loguru.Logger
) -> None:
  """Run socket server."""
  SmatterRequestHandler.input_queue = input_queue
  SmatterRequestHandler.output_queue = output_queue
  SmatterRequestHandler.logger = _logger
  with TCPServer((host, port), SmatterRequestHandler) as server:
    logger.info("Listening on {host}:{port}", host=host, port=port)
    server.serve_forever()

class SmatterRequestHandler(BaseRequestHandler):
  """Handle audio data stream from client."""

  logger: loguru.Logger
  input_queue: mp.Queue
  output_queue: mp.Queue
  buffer_size: int = 1024

  def setup(self) -> None:
    """Setup connection."""
    super().setup()
    if SmatterRequestHandler.input_queue is None or SmatterRequestHandler.output_queue is None:
      raise ValueError("Missing input queue, request handler cannot function.")
    if self.logger is None:
      SmatterRequestHandler.logger = loguru.logger
    SmatterRequestHandler.logger.info("Connection from {ca}.", ca=self.client_address)

  def finish(self) -> None:
    """Finish connection."""
    super().finish()
    SmatterRequestHandler.logger.info("Connection from {ca} closed.", ca=self.client_address)

  @staticmethod
  def __read_buffered(_socket: socket.socket, client_address):
    data = b''
    while True:
      while len(data) < SmatterRequestHandler.buffer_size:
        chunk = _socket.recv(SmatterRequestHandler.buffer_size - len(data))
        if not chunk:
          SmatterRequestHandler.logger.info(
            "Socket connection from {ca} returned no data.",
            ca=client_address
          )
          break
        data += chunk
      if len(data) == SmatterRequestHandler.buffer_size:
        yield data
      elif len(data) > SmatterRequestHandler.buffer_size:
        yield data[:SmatterRequestHandler.buffer_size]
        data = data[SmatterRequestHandler.buffer_size:]
      else:
        yield b''
        break
      data = b''

  @staticmethod
  def __write_result(_socket: socket.socket, client_address, closed: threading.Event):
    try:
      while True:
        try:
          while result := SmatterRequestHandler.output_queue.get_nowait():
            if closed.is_set() or result is None:
              break
            start, end, tx = result
            _socket.sendall(f'{start}-{end}: {tx or "<silence>"}'.encode())
        except queue.Empty:
          if closed.is_set():
            break
    except ConnectionAbortedError:
      pass
    except OSError:
      pass
    except Exception:
      SmatterRequestHandler.logger.exception(
        "Unexpected exception writing to connection from {ca}.",
        ca=client_address
      )
    SmatterRequestHandler.logger.info(
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
        self.request,
        self.client_address,
        self.close_event
      ),
      name=f"smatter_socker_write_{self.client_address}",
      daemon=True
    )
    self.thread.start()
    try:
      for data in SmatterRequestHandler.__read_buffered(
          self.request,
          self.client_address):
        if data:
          SmatterRequestHandler.input_queue.put(data)
    except ConnectionAbortedError:
      SmatterRequestHandler.logger.info(
        "Socket connection from {ca} aborted.",
        ca=self.client_address
      )
    except Exception:
      SmatterRequestHandler.logger.exception(
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
  s.settimeout(10.0)
  while output := s.recv(1024):
    _logger.info('Server returned {output}', output=output)
  s.close()

if __name__ == "__main__":
  test_smatter_socket("localhost", 9999, logger)
