from __future__ import annotations
import io
import multiprocessing as mp, threading as th, time, loguru, re
from multiprocessing.synchronize import Event
from multiprocessing.connection import PipeConnection
from typing import IO, TextIO, Tuple
from loguru import logger
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


def fix_elapsed(record: loguru.Record):
   """
   Fixes the elapsed value in log messages to display appropriately
   """
   record["extra"]["elapsed"] = str(record["elapsed"])

# Primary logger, kept global for easy access
__logger: loguru.Logger | None

def setup_logger(log_level: str):
  """
  Set up loguru logger for use in multiprocessing
  """
  global __logger
  logging_redirect_tqdm()
  logger.remove()
  if log_level != 'NONE':
    logger.configure(patcher=fix_elapsed)
    logger.add(sink=lambda m: tqdm.write(m.strip()), enqueue=True, colorize=True, level=log_level, backtrace=True, diagnose=True, format="<g>{extra[elapsed]}</g> | <level>{level: <8}</level> | <c>{process.name}</c>:<c>{thread.name}</c>:<c>{process.id}</c>:<c>{function}</c>:<c>{line}</c> - <level>{message}</level>")
    logger.warning(f'Setting smatter log level to {log_level}')
  __logger = logger

def get_logger() -> loguru.Logger:
  """
  Method to get the global logger for use in threads of multiprocessing
  """
  global __logger
  if not __logger:
    setup_logger('INFO')
  return __logger # type: ignore

def close_queue(queue: mp.Queue, stop: Event, _logger: loguru.Logger, close_fast = False):
  try:
    queue.put(None)
    while not close_fast and not stop.is_set() and not queue.empty():
      _logger.debug(f'Waiting for {queue.qsize()} items to be processed.')
      time.sleep(0.1)
    _logger.info(f'Closing output queue, state is stop: [{stop.is_set()}], close fast: [{close_fast}], queue size: [{queue.qsize()}]')
    queue.close()
    queue.join_thread()
  except Exception as e:
    _logger.exception(e)

def hms_match(s: str):
  """
  Match an hour:minutes:seconds string
  """
  return re.fullmatch(r"(?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d)", s)

def pipe_to_pipe(
    stop: Event, 
    _logger: loguru.Logger, 
    name: str, 
    buffer_size: int, 
    pipe_in: IO[bytes], 
    pipe_out: IO[bytes]):
  """
  Threads a function that feeds from one pipe into another
  This maybe be removable if it becomes clear how to pipe
  two processes together using subprocess + the ffmpeg lib
  """
  def feed():
    try:
      while not stop.is_set() and (buffer:= pipe_in.read(buffer_size)):
        pipe_out.write(buffer)
    except BrokenPipeError:
      _logger.error('Broken pipe for {n}, thread closing.', n=name)
    except Exception as e:
      _logger.exception(e)
    finally:
      _logger.warning(f'Now closing {name} thread, ideally at the end of reading the stream.')
    return
  
  return th.Thread(name=name, target=feed, daemon=True)

def pipe_to_mp_queue(
    stop: Event, 
    _logger: loguru.Logger, 
    name: str, 
    buffer_size: int, 
    pipe_in: IO[bytes], 
    queue_out: mp.Queue):
  """
  Threads a function that feeds from one pipe into another
  This maybe be removable if it becomes clear how to pipe
  two processes together using subprocess + the ffmpeg lib
  """
  def feed():
    close_fast = False
    try:
      while not stop.is_set() and (buffer:= pipe_in.read(buffer_size)):
        queue_out.put(buffer)
    except BrokenPipeError:
      _logger.error('Broken pipe for {n}, thread closing.', n=name)
      close_fast = True
    except Exception as e:
      _logger.exception(e)
      close_fast = True
    finally:
      _logger.warning(f'Now closing {name} thread, ideally at the end of reading the stream.')
      close_queue(queue_out, stop, _logger, close_fast)
    return
  
  return th.Thread(name=name, target=feed, daemon=True)

def mp_queue_to_pipe(
    stop: Event,
    _logger: loguru.Logger,
    name: str,
    pipe_out: IO[bytes] | None,
    queue_in: mp.Queue,
    new_pipe: bool = False
  ) -> Tuple[th.Thread, IO[bytes]]:
  """
  Threads a function that feeds from an 
  mp queue into a pipe
  """
  if new_pipe or pipe_out is None:
    pipe_out = io.BytesIO()
  def feed():
    try:
      while not stop.is_set() and (buffer:= queue_in.get()):
        pipe_out.write(buffer)
    except BrokenPipeError:
      _logger.error('Broken pipe for {n}, thread closing.', n=name)
    except Exception as e:
      _logger.exception(e)
    return
    
  return th.Thread(name=name, target=feed, daemon=True), pipe_out

def ff_log_messages(stop: Event, _logger: loguru.Logger, pipe_in: TextIO):
  """
  Goal here is to ensure it's easy to see
  where logs are coming from, hence two
  identical functions for logging.
  """
  def ffmpeg_log():
    try:
      for line in pipe_in:
        _logger.debug(line.strip())
        if stop.is_set():
          break
    except Exception as e:
      _logger.exception(e)
  
  return th.Thread(name='ffmpeg_logging_thread', target=ffmpeg_log, daemon=True)

def ytdl_log_messages(stop: Event, _logger: loguru.Logger, pipe_in: TextIO):
  """
  Goal here is to ensure it's easy to see
  where logs are coming from, hence two
  identical functions for logging.
  """
  def ytdlp_log():
    try:
      for line in pipe_in:
        _logger.debug(line.strip())
        if stop.is_set():
          break
    except Exception as e:
      _logger.exception(e)
  
  return th.Thread(name='ytdlp_logging_thread', target=ytdlp_log, daemon=True)  

def pipe_split(
    stop: Event, 
    _logger: loguru.Logger, 
    name: str, 
    buffer_size: int,
    pipe_in: IO[bytes], 
    pipe_out: IO[bytes],
    passthrough_queue_out: mp.Queue):
  """
  Threads a function that feeds from one pipe into
  one output pipe, and one multiprocessing pipe.
  """
  def feed():
    close_fast = False
    try:
      while not stop.is_set() and (buffer:= pipe_in.read(buffer_size)):
        pipe_out.write(buffer)
        passthrough_queue_out.put(buffer)
    except BrokenPipeError:
      _logger.error('Broken pipe for {n}, thread closing.', n=name)
      close_fast = True
    except OSError as e:
      _logger.exception(e)
      _logger.error('MP pipe failure for {n}, thread closing.', n=name)
      close_fast = True
    except Exception as e:
      _logger.exception(e)
      close_fast = True
    finally:
      _logger.warning(f'Now closing {name} thread, ideally at the end of reading the stream.')
      close_queue(passthrough_queue_out, stop, _logger, close_fast)
    return
  
  return th.Thread(name=name, target=feed, daemon=True)