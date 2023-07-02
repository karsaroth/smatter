from __future__ import annotations
import loguru
import multiprocessing as mp
from pathlib import Path

# Just save the srt file. This might be a good single step for some people
def save_srt(_logger: loguru.Logger, srt_in: mp.Queue, dir: Path, srt_filename: str):
  srt_file = (dir / srt_filename).absolute().as_posix()
  _logger.info(f'Starting save_srt to {srt_file}')
  count = 0
  with open(srt_file, 'w', encoding="utf-8") as f:
    while (srt:= srt_in.get()):
      f.write(srt)
      f.flush()
      count += 1
      _logger.debug(f'Put {count} srt records into {srt_file}')
  _logger.info(f'End save_srt to {srt_file}')