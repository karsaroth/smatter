"""
The setup and monitoring code
for the smatter tool
"""


from __future__ import annotations
from multiprocessing.synchronize import Event
from multiprocessing.connection import PipeConnection
from typing import Callable, List, Literal, NoReturn, Tuple
from pathlib import Path
import time
import argparse
import os
import sys
import signal
import subprocess
import multiprocessing as mp
import threading as th
import loguru
from tqdm import tqdm
import smatter.utils as u
import smatter.transx as tx
import smatter.ff_process as ff
import smatter.media_out as mo
import smatter.stream_server as stream
from smatter.mpv_show import show_mpv_transx_window

def main():
  """
  Main smatter kick off point
  """

  #---------------------------------------------------------------------
  # PARSER SETUP
  #---------------------------------------------------------------------

  # Shared by most options
  common_parser = argparse.ArgumentParser(add_help=False)
  common_parser.add_argument("-d", "--output-dir",
    help="Directory to store output files",
    default="./output",
    type=str
  )
  common_parser.add_argument("-f", "--output-file",
    help="Filename for any output file",
    default="output.srt",
    type=str
  )
  common_parser.add_argument("goal",
    type=str,
    choices=[
      'translate',
      'transcribe'
    ],
    help="""Do you want to transcribe the source language,
         or translate it into English (can only translate
         into English currently, but transcription is 
         available for multiple languages)"""
  )
  common_parser.add_argument("source",
    help="""URL of the stream/video. You should be able to
            use any url supported by ytdlp, however, there
            may be some limitations based on what ffmpeg 
            can do with the source data.""",
    type=str
  )
  common_parser.add_argument("-l", "--source-language",
    type=str,
    default="en",
    help="""Source language short code (e.g. 'en' = English,
         'ja' = Japanese, 'de' = German). Default is 'en'.
         See ISO_639-1 for a complete list, however the
         Whisper model may not support all languages."""
  )
  common_parser.add_argument("-q", "--quality",
    type=str,
    default="best",
    help="""Max vertical video size (e.g 480 for 480p),
            or a ytdlp format string if you prefer more
            control. Default is ytdlp's 'best' format."""
  )
  common_parser.add_argument("-s", "--start",
    type=hms_check,
    default="0",
    help="""Start point of a vod in HH:mm:ss,
            to skip to a beginning point.
            Default is 00:00:00"""
  )

  # Main parser
  parser = argparse.ArgumentParser(
    prog="smatter",
    description="Watch a video with subtitles generated by a neural network.",
    epilog="For more information, visit https://github.com/karsaroth/smatter"
  )
  parser.add_argument(
    "-m", "--model-size", 
    type=str,
    default="large-v2",
    help="""Whisper model selection.
            By default this is the largest,
            best for translation, but it may
            be too slow for some systems,
            or require too much RAM or VRAM.""",
    choices=[
      "tiny",
      "base",
      "small",
      "medium",
      "large",
      "large-v2",
      "tiny.en",
      "base.en",
      "small.en",
      "medium.en"
    ]
  )
  parser.add_argument("-g", "--force-gpu",
    help="""Force using GPU for translation
            (requires CUDA libraries). Only
            necessary if you think it isn't
            already happening for some reason.""",
    action='store_true'
  )
  parser.add_argument("-c", "--cache-dir",
    help="""Cache for models and other data,
            this directory may grow quite large
            with model data if you use multiple
            model sizes. Even the default
            'large-v2' model is multiple GB in
            size. Default is './cache'""",
    default="./cache",
    type=str
  )
  parser.add_argument("-v", "--log-level",
    help="""How much log info to show in command
            window, can be useful to find issues,
            especially at 'debug', but can also 
            quickly get noisy and slow processes
            down.""",
    type=str,
    choices=[
      'debug', 
      'info', 
      'warning', 
      'error', 
      'critical', 
      'none'
    ],
    default='warning'
  )

  subparsers = parser.add_subparsers(
    help='How to output the subtitles:'
  )
  parser_srt = subparsers.add_parser('srt',
    help='Generate an SRT file',
    parents=[common_parser]
  )
  parser_srt.set_defaults(output='srt')

  parser_vtt = subparsers.add_parser('vtt',
    help='Generate a VTT file',
    parents=[common_parser]
  )
  parser_vtt.set_defaults(output='vtt')

  parser_stream = subparsers.add_parser('stream',
    help="""Restream a video with subtitles in a
            web player (available by default at
            http://localhost:9999/)"""
  )
  parser_stream.set_defaults(output='stream')
  parser_stream.add_argument("-p", "--stream-port",
    help="Port to host stream on",
    default=9999,
    type=int
  )
  parser_stream.add_argument("-i", "--stream-host",
    help="Host to bind stream to",
    default="localhost",
    type=str
  )
  parser_watch = subparsers.add_parser('watch',
    help='Watch a video with subtitles using MPV (requires separate MPV installation)',
    parents=[common_parser]
  )
  parser_watch.set_defaults(output='watch')

  args = Args()
  parser.parse_args(None, args)

  #---------------------------------------------------------------------
  # END
  #---------------------------------------------------------------------

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

  #
  # Process and Thread Monitoring
  #
  processes: List[mp.Process] = []
  subprocesses: List[subprocess.Popen] = []
  threads: List[th.Thread] = []

  # Process placeholders
  transx_process = None

  # Output file/folder setup
  cache_dir = Path(args.cache_dir)
  stream_output_dir = cache_dir / 'stream'

  # Read default gigo if available
  gigo_file = Path('./gigo.txt')
  gigo_phrases: List[str] = []
  if gigo_file.exists():
    _logger.info('Loading gigo phrases from gigo.txt')
    with gigo_file.open('r', encoding='UTF-8') as gigo:
      gigo_phrases = gigo.readlines()
  _logger.info('gigo phrases: {gigo}', gigo=gigo_phrases)

  default_lang = 'en'
  default_goal = 'transcribe'
  if hasattr(args, 'source_language'):
    default_lang = args.source_language
  if hasattr(args, 'goal'):
    default_goal = args.goal

  whisper_config = tx.WhisperConfig(
    model = 'faster_whisper',
    model_size = args.model_size,
    force_gpu = args.force_gpu,
    lang = default_lang,
    goal = default_goal,
    model_root = args.cache_dir,
    gigo_phrases=gigo_phrases
  )

  default_start = '0'
  default_source = ''
  if hasattr(args, 'start'):
    default_start = args.start
  if hasattr(args, 'source'):
    default_source = args.source

  transx_config = tx.TransXConfig(
    _logger = _logger,
    base_path = args.cache_dir,
    format = 'plain' if args.output == 'watch' or args.output == 'stream' else args.output,
    output_queue = transx_output_queue,
    requested_start = default_start,
    stop = stopper,
    stream_url = default_source,
    model_config=whisper_config,
  )

  _logger.debug('Checking caching files and folders')
  # Check a few prereqs
  if not (cache_dir).exists():
    cache_dir.mkdir()
  if 'stream' in args.output:
    if not (stream_output_dir).exists():
      stream_output_dir.mkdir()
  _logger.warning(
    '''Will download transx model to {cd} if it doesn't 
        exist or needs updating. This might take a long time!''',
    cd=cache_dir.as_posix()
  )
  tx.verify_and_prepare_models(_logger, transx_config['model_config'])
  _logger.warning('Transx model check complete.')

  if args.output == 'watch':
    #
    # Watch Process:
    #
    # Set up watch processes
    try:
      probed = ff.probe(_logger, args.source, args.quality)
      thumb_url = probed['thumbnail']
      name = probed['title']
      stream_config = ff.StreamInputConfig(
        url= args.source,
        start= args.start,
        quality= args.quality,
        cache_dir= args.cache_dir,
      )
      stream_input = ff.MultiprocessStreamInput(_logger, stream_config, log_level == 'DEBUG')
      stream_input.start()

      tx_piped_args: Tuple[tx.TransXConfig, tx.WhisperConfig, mp.Queue] = (
        transx_config, whisper_config, stream_input.pcm_queue
      )
      transx_process = mp.Process(target=tx.transx_from_queue, args=tx_piped_args)
      transx_process.start()
      processes.append(transx_process)
      subs = live_bar_update_fun(
        tqdm(desc='Subtitles Available', total=25, unit='subtitle', ),
        transx_output_queue.qsize
      )
      pcm = reverse_live_bar_update_fun(
        tqdm(desc='PCM Backlog', total=100, unit='chunk', ),
        stream_input.pcm_queue.qsize
      )
      passthrough = live_bar_update_fun(
        tqdm(desc='Video Backlog', total=100, unit='chunk', ),
        stream_input.passthrough_queue.qsize
      )
      status_bar_thread = thread_status_bars(stopper, [subs, pcm, passthrough])
      threads.append(status_bar_thread)
      log_or_print('Close MPV window to end the program')
      mpv_thread = th.Thread(
        target=show_mpv_transx_window,
        args=(
          stopper,
          _logger,
          transx_output_queue,
          stream_input.passthrough_queue,
          thumb_url,
          name
        )
      )
      mpv_thread.start()
      threads.append(mpv_thread)

    except Exception as ex:
      _logger.exception(ex)
      stopper.set()
  elif args.output in ('srt', 'vtt'):
    #
    # SRT (Save file) Thread:
    #
    # Set up SRT process
    try:
      _logger.debug('Creating save_srt task')
      output_dir = Path(args.output_dir)
      if not output_dir.exists():
        output_dir.mkdir()
      if (output_dir / args.output_file).exists():
        log_or_print(
          '''Output directory must be cleared of previous files, 
          or the output filename should be adjusted.'''
        )
        return
      tx_out_args: Tuple[tx.TransXConfig, tx.WhisperConfig, None] = (
        transx_config, whisper_config, None
      )
      transx_process = mp.Process(target=tx.transx_from_audio_stream, args=tx_out_args)
      transx_process.start()
      processes.append(transx_process)
      save_thread = th.Thread(
        target=mo.save_srt,
        args=(stopper, _logger, transx_output_queue,
              args.output == 'vtt', output_dir, args.output_file)
      )
      save_thread.start()
      threads.append(save_thread)
      subs = live_bar_update_fun(
        tqdm(desc='Subtitles Waiting', total=25, unit='subtitle', ), transx_output_queue.qsize
      )
      status_bar_thread = thread_status_bars(stopper, [subs])
    except Exception as ex:
      _logger.exception(ex)
      stopper.set()
  elif args.output == 'stream':
    #
    # Streaming Thread:
    #
    # Set up HLS streaming process
    #
    # Control is handled via http requests
    try:
      server_thread = stream.run_server(
        _logger,
        args.stream_host,
        args.stream_port,
        wwwpath,
        stream_output_dir,
        transx_config
      )
      threads.extend([server_thread])

    except Exception as ex:
      _logger.exception(ex)
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
  """
  A function to remove closed processes and threads
  from the tracking lists
  """
  def check_closed(process: mp.Process):
    try:
      return process.is_alive()
    except ValueError:
      return False

  processes = [p for p in processes if check_closed(p)]
  threads = [t for t in threads if t.is_alive()]
  subprocesses = [sp for sp in subprocesses if sp.poll() is None]
  return processes, threads, subprocesses

def terminate_and_join(
    _logger: loguru.Logger,
    processes: List[mp.Process],
    threads: List[th.Thread],
    subprocesses: List[subprocess.Popen]
  ):
  """
  A cleanup function for processes and threads
  """
  for process in processes:
    try:
      process.terminate()
      time.sleep(0.3)
      process.kill()
      if process.pid:
        os.kill(process.pid, signal.SIGINT)
      process.close()
    except Exception as ex:
      _logger.exception(ex)
  for _subprocess in subprocesses:
    try:
      _subprocess.terminate()
      time.sleep(0.3)
      _subprocess.kill()
      if _subprocess.pid:
        os.kill(_subprocess.pid, signal.SIGINT)
    except Exception as ex:
      _logger.exception(ex)
  for thread in threads:
    if thread and thread.is_alive():
      thread.join(0.5)


def handle_main(
    processes: List[mp.Process],
    subprocesses: List[subprocess.Popen],
    threads: List[th.Thread],
    stopper: Event,
    _logger: loguru.Logger
  ):
  """
  Monitoring for threads and processes
  that have been created and need to be
  run to their conclusion and cleaned up
  if possible
  """
  open_p, open_t, open_sp = update_tracking(processes, threads, subprocesses)
  try:
    while not stopper.is_set() and (
          len(open_p) > 0 or
          len(open_t) > 0 or
          len(open_sp) > 0
      ):
      for process in filter(lambda p: not p.is_alive(), open_p):
        _logger.log(
          'ERROR' if process.exitcode != 0 else 'INFO',
          'Process {name} exited with code {exitcode}',
          name = process.name,
          exitcode = process.exitcode
        )
      for _subprocess in filter(lambda sp: sp.poll() is not None, open_sp):
        sp_arg0 = _subprocess.args[0] if isinstance(_subprocess.args, list) \
                  else str(_subprocess.args)
        _logger.log(
          'ERROR' if _subprocess.returncode != 0 else 'INFO',
          'Subprocess {name} exited with code {exitcode}',
          name = sp_arg0,
          exitcode = _subprocess.returncode
        )
        if _subprocess.returncode != 0 and sp_arg0 == 'yt-dlp':
          _logger.critical('yt-dlp failed, considering this a fatal error and shutting down.')
          stopper.set()
      for _thread in filter(lambda t: not t.is_alive(), open_t):
        _logger.info(f'Thread {_thread.name} exited.')
      open_p, open_t, open_sp = update_tracking(open_p, open_t, open_sp)
      _logger.info(
        '{p} processes, {sp} subprocesses, and {t} threads still running', 
        p=len(open_p), sp=len(open_sp), t=len(open_t)
      )
      time.sleep(5)
  except (KeyboardInterrupt, SystemExit):
    _logger.warning('Exit Requested, will attempt to clean up and close gracefully')
  except Exception as ex:
    _logger.exception(ex)
  finally:
    stopper.set()
    time.sleep(0.5)
    open_p, open_t, open_sp = update_tracking(open_p, open_t, open_sp)
    terminate_and_join(_logger, open_p, open_t, open_sp)
  _logger.info('All done')
  sys.exit()

class Args(argparse.Namespace):
  """
  Typing for the args generated from parseargs
  """
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

def hms_check(candidate: str):
  """
  Check that a string is a valid HH:mm:ss time
  """
  if u.hms_match(candidate):
    return candidate
  else:
    raise argparse.ArgumentTypeError(f"--start {candidate} invalid. Must be HH:mm:ss")

def reverse_live_bar_update_fun(progress_bar: tqdm[NoReturn], val: Callable[[], int]):
  """
  Create a function that will 
  update a progress bar based 
  on a value that is expected 
  to decrease when things are
  going well
  """
  def update():
    try:
      new_val = val()
      if progress_bar.n < new_val:
        if new_val > 1000:
          progress_bar.colour = 'red' if new_val > 10000 else 'yellow'
      elif progress_bar.n > new_val:
        progress_bar.colour = 'green'
      if progress_bar.total < new_val:
        progress_bar.total = new_val
      progress_bar.n = new_val
      progress_bar.refresh()
      return True
    except Exception:
      progress_bar.clear()
      progress_bar.close()
      return False
  return update, progress_bar

def live_bar_update_fun(progress_bar: tqdm[NoReturn], val: Callable[[], int]):
  """
  Create a function that will
  update a progress bar based
  on a value that is expected
  to increase when things are
  going well
  """

  def update():
    try:
      new_val = val()
      if progress_bar.n > new_val:
        if new_val < 10:
          progress_bar.colour = 'red' if new_val < 3 else 'yellow'
      elif progress_bar.n < new_val:
        progress_bar.colour = 'green'
      if progress_bar.total < new_val:
        progress_bar.total = new_val
      progress_bar.n = new_val
      progress_bar.refresh()
      return True
    except Exception:
      progress_bar.clear()
      progress_bar.close()
      return False
  return update, progress_bar

def update_all_bars(bars: List[Tuple[Callable[[], bool], tqdm[NoReturn,]]]):
  """
  Create a function that will
  update a list of progress bars
  based on their update functions
  """
  def update():
    status_track: dict[Tuple[Callable[[], bool], tqdm[NoReturn,]], bool] = \
      dict((_bar, True) for _bar in bars)
    for _bar, status in status_track.items():
      if status:
        status_track[_bar] = _bar[0]()
    return any(status_track.values())
  return update

def thread_status_bars(stopper: Event, bars: List[Tuple[Callable[[], bool], tqdm[NoReturn,]]]):
  """
  Run a thread that will update
  a list of progress bars until
  the stopper event is set
  """
  def run_until_all_bars_closed():
    while not stopper.is_set() and update_all_bars(bars):
      time.sleep(0.2)
    for _bar in bars:
      _bar[1].clear()
      _bar[1].close()
  thread = th.Thread(target=run_until_all_bars_closed, name='status_bars', daemon=True)
  thread.start()
  return thread


if __name__ == "__main__":
  main()
