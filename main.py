from __future__ import annotations
import asyncio
from multiprocessing.synchronize import Event
from multiprocessing.connection import PipeConnection
from tqdm import tqdm
from typing import Callable, List, Literal, NoReturn, Tuple, TypedDict
from pathlib import Path
import time
import argparse
import loguru
import os
import sys
import signal
import subprocess
import multiprocessing as mp
import threading as th
import smatter.utils as u
import smatter.transx as tx
import smatter.ff_process as ff
import smatter.media_out as mo
import smatter.stream_server as stream
from smatter.mpv_show import show_mpv_transx_window

def main():
  parser = argparse.ArgumentParser(
    prog="smatter",
    description="Watch a video with subtitles generated by a neural network.",
    epilog="For more information, visit https://github.com/karsaroth/smatter"
  )
  parser.add_argument("output", type=str, choices=['srt', 'vtt', 'watch', 'stream'], help="What output format is desired. 'srt' or 'vtt' will generate files for later use with plugins or video players. 'stream' will host a video stream with subtitles included, available by default at http://localhost:9999/. 'watch' will attempt to open an MPV window (MPV required).")
  parser.add_argument("goal", type=str, choices=['translate', 'transcribe'], help="Do you want to transcribe the source language, or translate it into English (can only translate into English currently, but transcription is available for multiple languages)", )
  parser.add_argument("source", help="URL of the stream/video. You should be able to use any url supported by ytdlp, however, there may be some limitations based on what ffmpeg can do with the source data.", type=str)
  parser.add_argument("-l", "--source-language", type=str, default="en", help="Source language short code (e.g. 'en' = English, 'ja' = Japanese, 'de' = German). Default is 'en'. See ISO_639-1 for a complete list, however the Whisper model may not support all languages.",)
  parser.add_argument("-q", "--quality", type=str, default="best", help="Max vertical video size (e.g 480 for 480p), or a ytdlp format string if you prefer more control. Default is ytdlp's 'best' format.")
  parser.add_argument("-s", "--start", type=hms_check, default="0", help="Start point of a vod in HH:mm:ss, to skip to a beginning point. Default is 00:00:00")
  parser.add_argument("-m", "--model-size", type=str, default="large-v2", help="Whisper model selection. By default this is the largest, best for translation, but it may be too slow for some systems, or require too much RAM or VRAM.",
                      choices=["tiny", "base", "small", "medium", "large", "large-v2", "tiny.en", "base.en", "small.en", "medium.en"])
  parser.add_argument("-g", "--force-gpu", help="Force using GPU for translation (requires CUDA libraries). Only necessary if you think it isn't already happening for some reason.", action='store_true')
  parser.add_argument("-c", "--cache-dir", help="Cache for models and other data, this directory may grow quite large with model data if you use multiple model sizes. Even the default 'large-v2' model is multiple GB in size. Default is './cache'", default="./cache", type=str)
  parser.add_argument("-v", "--log-level", help="How much log info to show in command window, can be useful to find issues, especially at 'debug', but can also quickly get noisy and slow processes down.", type=str, choices=['debug', 'info', 'warning', 'error', 'critical', 'none'], default='warning')

  out_args = parser.add_argument_group("Output options", "Options for 'srt' or 'vtt' files",)
  out_args.add_argument("-d", "--output-dir", help="Directory to store output files", default="./output", type=str)
  out_args.add_argument("-f", "--output-file", help="Filename for any output file", default="output.srt", type=str)

  stream_args = parser.add_argument_group("Streaming options", "Options for 'stream' output",)
  stream_args.add_argument("-p", "--stream-port", help="Port to host stream on", default=9999, type=int)
  stream_args.add_argument("-i", "--stream-host", help="Host to bind stream to", default="localhost", type=str)

  args = Args()
  parser.parse_args(None, args)
  #Prep external tools:
  basepath = os.path.dirname(os.path.abspath(__file__))
  binspath = os.path.join(basepath, 'libs/bin')
  wwwpath = Path(os.path.join(basepath, 'www'))
  os.environ['PATH'] = binspath + os.pathsep + os.pathsep + os.environ['PATH']

  #Kick off work
  log_level = args.log_level.upper()
  u.setup_logger(log_level)
  _logger = u.get_logger()
  def log_or_print(msg: str, **kwargs):
    if log_level == 'NONE':
      print(msg.format(**kwargs))
    _logger.warning(msg, **kwargs)

  # Stop signal
  stopper = mp.Event()

  #
  # Queues
  #
  # Subtitle/Caption Output
  transx_output_queue = mp.Queue()
  # Unmodified audio/video from ytdlp
  passthrough_queue = mp.Queue()
  # Audio from ytdlp in PCM format for whisper
  pcm_queue = mp.Queue()

  #
  # Process and Thread Monitoring
  #
  processes: List[mp.Process] = []
  subprocesses: List[subprocess.Popen] = []
  threads: List[th.Thread] = []

  # Process placeholders
  transx_process = None
  ff_process = None
  ytdl_process = None
  hls_process = None

  # Output file/folder setup
  output_dir = Path(args.output_dir)
  stream_output_dir = output_dir / 'stream'
  cache_dir = Path(args.cache_dir)

  transx_config: tx.TransXConfig = {
    '_logger': _logger,
    'base_path': args.cache_dir,
    'format': 'plain' if args.output == 'watch' or args.output == 'stream' else args.output,
    'output_queue': transx_output_queue,
    'requested_start': args.start,
    'stop': stopper,
    'stream_url': args.source,
  }
  whisper_config: tx.WhisperConfig = {
    'model_size': args.model_size,
    'force_gpu': args.force_gpu,
    'lang': args.source_language,
    'goal': args.goal,
    'model_root': args.cache_dir,
  }
  
  # We have to save some output files, ensure directories
  # are available
  if args.output in ['srt', 'vtt', 'stream']:
    _logger.debug('Checking output files and folders')
    # Check a few prereqs
    if not output_dir.exists():
      output_dir.mkdir()
    if 'stream' in args.output:
      if not (stream_output_dir).exists():
        stream_output_dir.mkdir()
    if not (cache_dir).exists():
      cache_dir.mkdir()
    _logger.warning('Will download transx model to {cd} if it doesn\'t exist or needs updating. This might take a long time!', cd=cache_dir.as_posix())
    tx.check_and_download(whisper_config['model_size'], whisper_config['model_root'])
    _logger.warning('Transx model check complete.')

    if ((args.output == 'srt' or args.output == 'vtt') and \
        (output_dir / args.output_file).exists()):

      log_or_print('Output directory must be cleared of previous files, or the output filename should be adjusted.')
      return

  if args.output == 'watch':
    #
    # Watch Process:
    #
    # Set up watch processes
    subs = live_bar_update_fun(tqdm(desc='Subtitles Available', total=25, unit='subtitle', ), transx_output_queue.qsize)
    pcm = reverse_live_bar_update_fun(tqdm(desc='PCM Backlog', total=100, unit='chunk', ), pcm_queue.qsize)
    passthrough = live_bar_update_fun(tqdm(desc='Video Backlog', total=100, unit='chunk', ), passthrough_queue.qsize)
    try:
      probed = ff.probe(_logger, args.source, args.quality)
      thumb_url = probed['thumbnail']
      name = probed['title']
      ytdl_process, ytdl_log_thread = ff.url_into_pipe(
        stopper, 
        _logger, 
        log_level,
        args.cache_dir, 
        args.source, 
        args.start, 
        args.quality
      )
      subprocesses.append(ytdl_process)
      if ytdl_log_thread:
        threads.append(ytdl_log_thread)
      
      ff_process, feed_thread, pcm_feed_thread, ff_log_thread = ff.pipe_into_mp_queue(
        stopper, 
        _logger, 
        args.log_level == 'debug', 
        ytdl_process, 
        pcm_queue, 
        passthrough_queue
      )
      subprocesses.append(ff_process)
      threads.extend([feed_thread, pcm_feed_thread])
      if ff_log_thread:
        threads.append(ff_log_thread)

      tx_piped_args: Tuple[tx.TransXConfig, tx.WhisperConfig, mp.Queue] = (transx_config, whisper_config, pcm_queue)
      transx_process = mp.Process(target=tx.transx_from_queue, args=tx_piped_args)
      transx_process.start()
      processes.append(transx_process)
      status_bar_thread = thread_status_bars(stopper, [subs, pcm, passthrough])
      threads.append(status_bar_thread)
      log_or_print('Close MPV window to end the program')
      mpv_thread = th.Thread(target=show_mpv_transx_window, args=(stopper, _logger, transx_output_queue, passthrough_queue, thumb_url, name))
      mpv_thread.start()
      threads.append(mpv_thread)      

    except Exception as e:
      _logger.exception(e)
      stopper.set()
  elif args.output == 'srt' or args.output == 'vtt':
    #
    # SRT (Save file) Thread:
    #
    # Set up SRT process
    try:
      _logger.debug('Creating save_srt task')
      tx_out_args: Tuple[tx.TransXConfig, tx.WhisperConfig, None] = (transx_config, whisper_config, None)
      transx_process = mp.Process(target=tx.transx_from_audio_stream, args=tx_out_args)
      transx_process.start()
      processes.append(transx_process)
      save_thread = th.Thread(target=mo.save_srt, args=(stopper, _logger, transx_output_queue, args.output == 'vtt', output_dir, args.output_file))
      save_thread.start()
      threads.append(save_thread)
      subs = live_bar_update_fun(tqdm(desc='Subtitles Waiting', total=25, unit='subtitle', ), transx_output_queue.qsize)
      pcm = reverse_live_bar_update_fun(tqdm(desc='PCM Waiting', total=100, unit='chunk', ), pcm_queue.qsize)
      status_bar_thread = thread_status_bars(stopper, [subs, pcm])
    except Exception as e:
      _logger.exception(e)
      stopper.set()

  elif args.output == 'stream':
    #
    # Streaming Thread:
    #
    # Set up HLS streaming process
    try:
      # _logger.info('Checking for stream details')
      # probed = ff.probe(_logger, args.source, args.quality)
      ytdl_process, ytdl_log_thread = ff.url_into_pipe(
        stopper, 
        _logger,
        log_level,
        args.cache_dir,
        args.source, 
        args.start, 
        args.quality
      )
      subprocesses.append(ytdl_process)
      if ytdl_log_thread:
        threads.append(ytdl_log_thread)

      ff_process, feed_thread, pcm_feed_thread, ff_log_thread = ff.pipe_into_mp_queue(
        stopper, 
        _logger, 
        args.log_level == 'debug', 
        ytdl_process, 
        pcm_queue, 
        passthrough_queue
      )
      subprocesses.append(ff_process)
      threads.extend([feed_thread, pcm_feed_thread])
      if ff_log_thread:
        threads.append(ff_log_thread)

      tx_piped_args: Tuple[tx.TransXConfig, tx.WhisperConfig, mp.Queue] = (transx_config, whisper_config, pcm_queue)
      transx_process = mp.Process(target=tx.transx_from_queue, args=tx_piped_args)
      transx_process.start()
      processes.append(transx_process)
      
      hls_process, hls_feed_thread, hls_log_thread = ff.mp_queue_into_hls_stream(
        stopper,
        _logger,
        stream_output_dir,
        args.log_level == 'debug',
        passthrough_queue,
        6
      )
      subprocesses.append(hls_process)
      threads.extend([hls_feed_thread])
      if hls_log_thread:
        threads.append(hls_log_thread)

      subs = live_bar_update_fun(tqdm(desc='Subtitles Backlog', total=25, unit='subtitle', ), transx_output_queue.qsize)
      pcm = reverse_live_bar_update_fun(tqdm(desc='PCM Backlog', total=100, unit='chunk', ), pcm_queue.qsize)
      passthrough = live_bar_update_fun(tqdm(desc='Video Backlog', total=100, unit='chunk', ), passthrough_queue.qsize)
      status_bar_thread = thread_status_bars(stopper, [subs, pcm, passthrough])
      server_thread = stream.run_server(_logger, args.stream_host, args.stream_port, wwwpath, stream_output_dir, transx_output_queue)
      threads.extend([status_bar_thread, server_thread])

    except Exception as e:
      _logger.exception(e)
      stopper.set()
       
  else:
    log_or_print('Not currently implemented')

  _logger.debug('Starting monitoring of P/SP/T in main thread')
  handle_main(processes, subprocesses, threads, stopper, _logger)
  _logger.debug('Main thread exiting')

def update_tracking(
    processes: List[mp.Process], 
    threads: List[th.Thread], 
    subprocesses: List[subprocess.Popen]
    ):
  def check_closed(process: mp.Process):
    try:
      return process.is_alive()
    except ValueError:
      return False
    
  processes = [p for p in processes if check_closed(p)]
  threads = [t for t in threads if t.is_alive()]
  subprocesses = [sp for sp in subprocesses if sp.poll() is None]
  return processes, threads, subprocesses

def subprocess_first_arg(subprocess: subprocess.Popen):
  return subprocess.args[0] if isinstance(subprocess.args, list) else str(subprocess.args)

def terminate_and_join(
    _logger: loguru.Logger,
    processes: List[mp.Process], 
    threads: List[th.Thread], 
    subprocesses: List[subprocess.Popen],
    kill: bool
  ):
  for p in processes:
    if p and p.is_alive():
      if kill:
        if os.name == 'posix':
          p.kill()
        else:
          if not p.pid:
            _logger.warning('Process {p} has no PID', p.name)
          else:
            os.kill(p.pid, signal.SIGINT)
        p.join(0.5)
        p.close()
      else:
        p.terminate()
  for sp in subprocesses:
    if sp and sp.poll() is None:
      if kill:
        if os.name == 'posix':
          sp.kill()
        else:
          if not sp.pid:
            _logger.warning('Process {p} has no PID, cannot signal it to stop.', subprocess_first_arg(sp))
          else:
            os.kill(sp.pid, signal.SIGINT)
      else:
        sp.terminate()
  for t in threads:
    if t and t.is_alive():
      t.join(0.5)
  

def handle_main(
    processes: List[mp.Process], 
    subprocesses: List[subprocess.Popen], 
    threads: List[th.Thread],
    stopper: Event, 
    _logger: loguru.Logger
  ):
  open_p, open_t, open_sp = update_tracking(processes, threads, subprocesses)
  try:
    while not stopper.is_set() and (
          len(open_p) > 0 or 
          len(open_t) > 0 or
          len(open_sp) > 0
      ):
      for p in open_p:
        if not p.is_alive():
          if p.exitcode != 0:
            _logger.error(f'Process {p.name} exited with code {p.exitcode}')
          else:
            _logger.info(f'Process {p.name} exited normally')
      for sp in open_sp:
        if sp.poll() is not None:
          if sp.returncode != 0:
            sp_arg0 = subprocess_first_arg(sp)
            _logger.error(f'Subprocess {sp_arg0} exited with code {sp.returncode}')
            if sp_arg0 == 'yt-dlp':
              _logger.critical('yt-dlp failed, considering this a fatal error and shutting down.')
              stopper.set()
          else:
            _logger.info(f'Subprocess {sp.args} exited normally')
      for t in open_t:
        if not t.is_alive():
          _logger.info(f'Thread {t.name} exited.')
      open_p, open_t, open_sp = update_tracking(open_p, open_t, open_sp)
      _logger.info('{p} processes, {sp} subprocesses, and {t} threads still running', p=len(open_p), sp=len(open_sp), t=len(open_t))
      time.sleep(5)
  except KeyboardInterrupt or SystemExit:
    _logger.warning('Exit Requested, will attempt to clean up and close gracefully')
  except Exception as e:
    _logger.exception(e)
  finally:
    stopper.set()
    time.sleep(0.5)
    open_p, open_t, open_sp = update_tracking(open_p, open_t, open_sp)
    terminate_and_join(_logger, open_p, open_t, open_sp, False)
    try:
      open_p, open_t, open_sp = update_tracking(open_p, open_t, open_sp)
      terminate_and_join(_logger, open_p, open_t, open_sp, True)
    except Exception as e:
      _logger.exception(e)
      _logger.warning('Fast dropping main thread, some processes may not have been cleaned up properly.')
  _logger.info('All done')
  sys.exit()

class Args(argparse.Namespace):
  source: str
  quality: str
  start: str
  output: Literal['srt', 'vtt', 'watch']
  output_dir: str
  output_file: str
  model_size: str
  force_gpu: bool
  source_language: str
  goal: Literal['translate', 'transcribe']
  log_level: Literal['debug', 'info', 'warning', 'error', 'critical', 'none']
  cache_dir: str
  stream_port: int
  stream_host: str

def hms_check(s: str):
  if u.hms_match(s):
    return s
  else:
    raise argparse.ArgumentTypeError(f"--start {s} invalid. Must be HH:mm:ss")

def reverse_live_bar_update_fun(pb: tqdm[NoReturn], val: Callable[[], int]):
  def update():
    try:
      new_val = val()
      if pb.n < new_val:
        if new_val > 1000:
          pb.colour = 'red' if new_val > 10000 else 'yellow'
      elif pb.n > new_val:
        pb.colour = 'green'
      if pb.total < new_val:
        pb.total = new_val
      pb.n = new_val
      pb.refresh()
      return True
    except Exception:
      pb.clear()
      pb.close()
      return False
  return update, pb

def live_bar_update_fun(pb: tqdm[NoReturn], val: Callable[[], int]):
  def update():
    try:
      new_val = val()
      if pb.n > new_val:
        if new_val < 10:
          pb.colour = 'red' if new_val < 3 else 'yellow'
      elif pb.n < new_val:
        pb.colour = 'green'
      if pb.total < new_val:
        pb.total = new_val
      pb.n = new_val
      pb.refresh()
      return True
    except Exception:
      pb.clear()
      pb.close()
      return False
  return update, pb

def update_all_bars(bars: List[Tuple[Callable[[], bool], tqdm[NoReturn,]]]):

  def update():
    status_track: dict[Tuple[Callable[[], bool], tqdm[NoReturn,]], bool] = dict((b, True) for b in bars)
    for b, status in status_track.items():
      if status:
        status_track[b] = b[0]()
    return any(status_track.values())
  return update

def thread_status_bars(stopper: Event, bars: List[Tuple[Callable[[], bool], tqdm[NoReturn,]]]):
  def run_until_all_bars_closed():
    while not stopper.is_set() and update_all_bars(bars):
      time.sleep(0.2)
    for b in bars:
      b[1].clear()
      b[1].close()
  thread = th.Thread(target=run_until_all_bars_closed, name='status_bars', daemon=True)
  thread.start()
  return thread


if __name__ == "__main__":
  main()