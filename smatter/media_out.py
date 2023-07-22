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
  srt_file = (dir / srt_filename).absolute().as_posix()
  _logger.info(f'Starting save_srt to {srt_file}')
  count = 0
  with open(srt_file, 'w', encoding="utf-8", newline='\n') as f:
    if vtt_header:
      f.write(f'WEBVTT\n\n')
      f.flush()
    while not stop.is_set() and (srt:= srt_in.get()):
      srt_record = srt
      f.write(srt_record)
      f.flush()
      count += 1
      _logger.debug(f'Put {count} vtt records into {srt_file}')
  _logger.info(f'End save_srt to {srt_file}')

def save_vtt_chunks(
    stop: Event,
    _logger: loguru.Logger,
    vtt_in: mp.Queue,
    length: int,
    dir: Path
  ):
  """
  Saves translation data to a series of files
  Which will be streamable alongside video and
  audio files generated by ffmpeg.
  """
  _logger.info(f'Starting save_vtt_chunks to {dir}')
  count = 0
  file_start = 0
  file_end = file_start + length
  file_name = f'stream_sub{count:06d}.webvtt'
  start = end = 0
  held_over = None
  pl = (dir / "stream_sub_vtt.m3u8").absolute().as_posix()
  with open(pl, "w", encoding="utf-8", newline='\n') as f:
    f.write(f"""#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:{length}
#EXT-X-MEDIA-SEQUENCE:0
""")
    f.flush()
  while not stop.is_set():
    vtt_file = (dir / file_name).absolute().as_posix()
    with open(vtt_file, 'w', encoding="utf-8", newline='\n') as f:
      f.write(f'WEBVTT\n\n')
      f.flush()
      if held_over is not None:
        start, end, held_over_record = held_over
        if start <= file_end:
          f.write(held_over_record)
          f.flush()
          held_over = None
      if not held_over or end <= file_end:
        while not stop.is_set() and (vtt:= vtt_in.get()):
          start, end, vtt_record = vtt
          if start > file_end:
            held_over = vtt
            break # go to next file
          f.write(vtt_record)
          f.flush()
          if end > file_end:
            break # go to next file
        else:
          break #finish creating files
    with open(pl, "a", encoding="utf-8", newline='\n') as f:
      f.write(f'#EXTINF:{end - file_start},\n')
      f.write(f'{file_name}\n')
      f.flush()
    file_start = file_end
    file_end = file_start + length
    count += 1
    file_name = f'stream_sub{count:06d}.webvtt'
  with open(pl, "a", encoding="utf-8", newline='\n') as f:
    f.write(f'#EXTINF:{end - file_start},\n')
    f.write(f'{file_name}\n')
    f.write(f'#EXT-X-ENDLIST\n')
    f.flush()
  _logger.info(f'End save_vtt_chunks to {dir}')