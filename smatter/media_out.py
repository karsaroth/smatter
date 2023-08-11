from __future__ import annotations
import loguru
import multiprocessing as mp
from pathlib import Path
from multiprocessing.synchronize import Event

def save_srt(
    stop: Event,
    _logger: loguru.Logger, 
    srt_in: mp.Queue,
    vtt_header: bool,
    dir: Path, 
    srt_filename: str
  ):
  """
  Saves translation data to a file.
  """
  try:
    srt_file = (dir / srt_filename).absolute().as_posix()
    _logger.info(f'Starting save_srt to {srt_file}')
    count = 0
    with open(srt_file, 'w', encoding="utf-8", newline='\n') as f:
      if vtt_header:
        f.write(f'WEBVTT\n\n')
        f.flush()
      while not stop.is_set() and (srt:= srt_in.get()):
        _, _, srt_record = srt
        f.write(srt_record)
        f.flush()
        count += 1
        _logger.debug(f'Put {count} vtt records into {srt_file}')
    _logger.info(f'End save_srt to {srt_file}')
  except KeyboardInterrupt or SystemExit:
    _logger.info(f'Keyboard interrupt or system exit, closing save_srt')
  except Exception as e:
    _logger.exception(e)
  finally:
    stop.set()