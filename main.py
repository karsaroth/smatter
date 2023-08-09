from __future__ import annotations
from multiprocessing.synchronize import Event
from multiprocessing.connection import PipeConnection
from tqdm import tqdm
from typing import Callable, List, Literal, NoReturn, Tuple, TypedDict
from pathlib import Path
import time
import argparse
import re
import loguru
import os
import multiprocessing as mp
import smatter.utils as u
import smatter.transx as tx
import smatter.ff_process as ff
import smatter.media_out as mo
import smatter.stream_server as stream
from smatter.mpv_show import show_mpv_transx_window

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--source", help="URL of stream/video", type=str, required=True)
  parser.add_argument("--quality", help="Max vertical video size (e.g 480 for 480p)", type=str, default="best")
  parser.add_argument("--start", help="Start point of a vod in HH:mm:ss", type=hms_check, default="0")
  parser.add_argument("--output", help="What output format is desired (file or video window)", type=str, choices=['srt', 'vtt', 'watch', 'stream'], required=True)
  parser.add_argument("--output-dir", help="Directory to store output files", default="./output", type=str, required=False)
  parser.add_argument("--output-file", help="Filename for any output file", default="output.srt", type=str, required=False)
  parser.add_argument("--model-size", help="Whisper model selection", type=str, default="base", required=False,
                      choices=["tiny", "base", "small", "medium", "large", "large-v2", "tiny.en", "base.en", "small.en", "medium.en"])
  parser.add_argument("--force-gpu", help="Force using GPU for translation (requires CUDA libraries). Only necessary if you think it isn't already happening for some reason.", action='store_true')
  parser.add_argument("--source-language", help="Source language short code", type=str, default="en", required=False)
  parser.add_argument("--goal", help="x in trans(x)", type=str, choices=['translate', 'transcribe'], default='transcribe', required=False)
  parser.add_argument("--log-level", help="How much log info to show in command window", type=str, choices=['debug', 'info', 'warning', 'error', 'critical', 'none'], default='warning', required=False)
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
  # For SRT output from Transx
  transx_output_queue = mp.Queue()
  passthrough_queue = mp.Queue()
  pcm_queue = mp.Queue()
  transx_process = None
  ff_process = None
  ytdl_process = None
  hls_process = None
  output_dir = Path(args.output_dir)
  stream_output_dir = output_dir / 'stream'
  # file_structs = re.match(r'^(.*?)(\.(.*))?$', args.output_file)

  transx_config: tx.TransXConfig = {
    '_logger': _logger,
    'base_path': './tmp',
    'format': 'mpv' if args.output == 'watch' else args.output,
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
  }
  
  # We have to save some output files
  if args.output in ['srt', 'vtt', 'stream']:
    _logger.debug('Checking output files and folders')
    # Check a few prereqs
    if not output_dir.exists():
      output_dir.mkdir()
    if 'stream' in args.output:
      if not (stream_output_dir).exists():
        stream_output_dir.mkdir()
    # if file_structs == None:
    #   log_or_print('Filename must have extension (e.g. ".mkv")')
    #   return

    if ((args.output == 'srt' or args.output == 'vtt') and (output_dir / args.output_file).exists()) or \
      (args.output == 'stream' and (output_dir / 'playlist.m3u8').exists()):
      log_or_print('Output directory must be cleared of previous files, or the output filename should be adjusted.')
      return
  if args.output == 'watch':
    #
    # Watch Process:
    #
    # Set up watch processes
    subs = live_bar_update_fun(tqdm(desc='Subtitles Available', total=25, unit='subtitle', ), transx_output_queue.qsize)
    pcm = reverse_live_bar_update_fun(tqdm(desc='PCM Data', total=100, unit='chunk', ), pcm_queue.qsize)
    passthrough = live_bar_update_fun(tqdm(desc='Video Data', total=100, unit='chunk', ), passthrough_queue.qsize)
    try:
      probed = ff.probe(_logger, args.source, args.quality)
      thumb_url = probed['thumbnail']
      name = probed['title']
      ytdl_process, _ytdl_log_thread = ff.url_into_pipe(
        stopper, 
        _logger if args.log_level == 'debug' else None, 
        './tmp', 
        args.source, 
        args.start, 
        args.quality
      )
      ff_process, _feed_thread, _pcm_feed_thread, _ff_log_thread = ff.pipe_into_mp_queue(
        stopper, 
        _logger, 
        args.log_level == 'debug', 
        ytdl_process, 
        pcm_queue, 
        passthrough_queue
      )
      tx_piped_args: Tuple[tx.TransXConfig, tx.WhisperConfig, mp.Queue] = (transx_config, whisper_config, pcm_queue)
      transx_process = mp.Process(target=tx.transx_from_queue, args=tx_piped_args)
      transx_process.start()
      log_or_print('Close MPV window to end the program')
      ufun = update_all_bars([subs, pcm, passthrough])
      show_mpv_transx_window(stopper, _logger, transx_output_queue, passthrough_queue, thumb_url, name, ufun)

    except Exception as e:
      _logger.exception(e)
    finally:
      stopper.set()
      _logger.info('Cleaning up')
      subs[1].clear()
      subs[1].close()
      pcm[1].clear()
      pcm[1].close()
      passthrough[1].clear()
      passthrough[1].close()
      time.sleep(0.5)
      if transx_process and transx_process.is_alive():
        transx_process.terminate()
      if ytdl_process and not ytdl_process.returncode:
        ytdl_process.terminate()
      if ff_process and not ff_process.returncode:
        ff_process.terminate()

      _logger.info('All done')
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
      mo.save_srt(stopper, _logger, transx_output_queue, args.output == 'vtt', output_dir, args.output_file)
    except Exception as e:
      _logger.exception(e)
    finally:
      _logger.info('Cleaning up')
      stopper.set()
      time.sleep(0.5)
      if transx_process and transx_process.is_alive():
        transx_process.terminate()
      _logger.info('All done')
  elif args.output == 'stream':
    #
    # Streaming Thread:
    #
    # Set up HLS streaming process
    try:
      _logger.debug('Checking for stream details')
      probed = ff.probe(_logger, args.source, args.quality)
      resoultion = probed['resolution']
      bandwidth = int(probed['tbr'] * 1000)
      ytdl_process, _ytdl_log_thread = ff.url_into_pipe(
        stopper, 
        _logger if args.log_level == 'debug' else None, 
        './tmp', 
        args.source, 
        args.start, 
        args.quality
      )
      ff_process, _feed_thread, _pcm_feed_thread, _ff_log_thread = ff.pipe_into_mp_queue(
        stopper, 
        _logger, 
        args.log_level == 'debug', 
        ytdl_process, 
        pcm_queue, 
        passthrough_queue
      )
      tx_piped_args: Tuple[tx.TransXConfig, tx.WhisperConfig, mp.Queue] = (transx_config, whisper_config, pcm_queue)
      transx_process = mp.Process(target=tx.transx_from_queue, args=tx_piped_args)
      transx_process.start()
      hls_process, _hls_feed_thread, _hls_log_thread = ff.mp_queue_into_hls_stream(
        stopper,
        _logger,
        stream_output_dir,
        args.log_level == 'debug',
        passthrough_queue,
        6,
        bandwidth,
        resoultion
      )
      # mo.save_vtt_chunks(
      #   stopper,
      #   _logger,
      #   transx_output_queue,
      #   6,
      #   stream_output_dir
      # )
      stream.run_server(_logger, 'localhost', 9999, wwwpath, stream_output_dir, transx_output_queue)
    except Exception as e:
      _logger.exception(e)
    finally:
      _logger.info('Cleaning up')
      stopper.set()
      time.sleep(0.5)
      if transx_process and transx_process.is_alive():
        transx_process.terminate()
      if ytdl_process and not ytdl_process.returncode:
        ytdl_process.terminate()
      if ff_process and not ff_process.returncode:
        ff_process.terminate()
      if hls_process and not hls_process.returncode:
        hls_process.terminate()
      _logger.info('All done')    
  else:
    log_or_print('Not currently implemented')

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


if __name__ == "__main__":
  main()