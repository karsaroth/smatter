from __future__ import annotations
from io import TextIOWrapper
from multiprocessing.synchronize import Event
import multiprocessing as mp
import queue
import threading as th
import subprocess
import json
import pathlib
import os
import time
import signal
import typing as ty
import abc
import ffmpeg as ff #type: ignore
import loguru
import smatter.utils as u

class StreamInputConfig(ty.TypedDict):
  """
  A dict containing the configuration
  for a stream input.
  """
  url: str
  start: str
  quality: str
  cache_dir: str

class SubprocessManager(abc.ABC):
  """
  Handle the subprocess
  and threads for certain
  AV transformations.
  """

  def __init__(self,
               _logger: loguru.Logger,
               debug = False):
    self.stopper = mp.Event()
    self.logger = _logger
    self.debug = debug
    self.subprocesses: ty.List[subprocess.Popen[bytes]] = []
    self.threads: ty.List[th.Thread] = []

  @abc.abstractmethod
  def start(self):
    """
    Starts the subprocesses and threads
    """

  @abc.abstractmethod
  def state_cleanup(self):
    """
    Cleans up any state that
    remains after stopping
    """

  @abc.abstractmethod
  def status_detail(self):
    """
    Returns a human readable
    string with status details
    """

  def is_running(self):
    """
    Returns True if any subprocesses
    or threads are still running.
    """
    return len(self.subprocesses) > 0 or len(self.threads) > 0

  def stop(self):
    """
    Sends a stop signal to all
    subprocesses and threads.
    """
    self.stopper.set()
    for _subprocess in self.subprocesses:
      try:
        _subprocess.terminate()
      except Exception as ex:
        self.logger.exception(ex)

  def reset(self):
    """
    Resets the stop signal
    """
    self.stopper.clear()

  def cleanup(self):
    """
    Cleans up the subprocesses and threads
    """
    for _subprocess in self.subprocesses:
      try:
        self.logger.info('Terminating subprocess {pid}', pid=_subprocess.pid)
        if _subprocess.poll() is None:
          _subprocess.send_signal(signal.SIGINT)
          time.sleep(0.3)
        if _subprocess.poll() is None:
          _subprocess.terminate()
          time.sleep(0.3)
        if _subprocess.poll() is None:
          _subprocess.kill()
          time.sleep(0.3)
        self.logger.info('Subprocess {pid} should now be terminated', pid=_subprocess.pid)
      except Exception as ex:
        self.logger.exception(ex)
    for thread in self.threads:
      if thread and thread.is_alive():
        thread.join(0.3)
    self.subprocesses = [sp for sp in self.subprocesses if sp.poll() is None]
    self.threads = [t for t in self.threads if t.is_alive()]
    self.logger.info('Subprocesses still active: {subprocesses}, Threads still active {threads}',
                     subprocesses=len(self.subprocesses), threads=len(self.threads))
    self.state_cleanup()

  def is_clean(self):
    """
    Returns True if all subprocesses
    and threads have been cleaned up.
    """
    return len(self.subprocesses) == 0 and len(self.threads) == 0

class MultiprocessStreamInput(SubprocessManager):
  """
  A class to represent a video
  stream input for use in 
  multiprocessing
  """
  def __init__(self,
               _logger: loguru.Logger,
               input_config: StreamInputConfig,
               debug = False):
    super().__init__(_logger, debug)
    self.input_config = input_config
    self.pcm_queue = mp.Queue()
    self.passthrough_queue = mp.Queue()

  def status_detail(self):
    if len(self.subprocesses) == 0:
      if len(self.threads) > 0:
        return 'Waiting/Blocked'
      return 'Stopped'
    if len(self.subprocesses) == 1:
      return 'Partially Running/Blocked'
    if len(self.subprocesses) == 2:
      if self.stopper.is_set():
        return 'Stopping/Blocked'
      stdout = self.subprocesses[0].stdout
      stdin = self.subprocesses[1].stdin
      if stdout:
        out_pipe_size = os.fstat(stdout.fileno()).st_size
      else:
        out_pipe_size = 0
      if stdin:
        in_pipe_size = os.fstat(stdin.fileno()).st_size
      else:
        in_pipe_size = 0
      return f'''yt| [{out_pipe_size}],
                 ff| [{in_pipe_size}], 
                 PCM Q [{self.pcm_queue.qsize()}]'''
    return 'Unknown/Unexpected State'

  def start(self):
    """
    Starts a stream input and returns
    a queue which will contain the stream
    data.
    """
    ytdl_process, ytdl_log_thread = url_into_pipe(
      self.stopper,
      self.logger,
      self.debug,
      self.input_config
    )
    self.subprocesses.append(ytdl_process)
    if ytdl_log_thread:
      self.threads.append(ytdl_log_thread)

    ff_process, feed_thread, pcm_feed_thread, ff_log_thread = pipe_into_mp_queue(
      self.stopper,
      self.logger,
      self.debug,
      ytdl_process,
      self.pcm_queue,
      self.passthrough_queue
    )
    self.subprocesses.append(ff_process)
    self.threads.extend([feed_thread, pcm_feed_thread])
    if ff_log_thread:
      self.threads.append(ff_log_thread)

  def state_cleanup(self):
    try:
      while not self.pcm_queue.empty():
        # Clean out the pcm queue
        self.pcm_queue.get_nowait()
    except (queue.Empty, ValueError):
      self.logger.info('PCM Queue empty/closed, clean complete.')
    try:
      while not self.passthrough_queue.empty():
        # Clean out the passthrough queue
        self.passthrough_queue.get_nowait()
    except (queue.Empty, ValueError):
      self.logger.info('Passthrough Queue empty/closed, clean complete.')

class MultiprocessStreamOutput(SubprocessManager):
  """
  Generates output data for
  a stream using ffmpeg, based
  on configuration.
  Currently only generating
  HLS stream data.
  """

  def __init__(self,
               _logger: loguru.Logger,
               base_dir: pathlib.Path,
               length: int,
               debug = False):
    super().__init__(_logger, debug)
    self.base_dir = base_dir
    self.length = length
    self.passthrough_queue = mp.Queue()
    # Ensure the base_dir exists
    if not self.base_dir.exists():
      self.base_dir.mkdir(parents=True, exist_ok=True)

  def __mp_queue_into_hls_stream(self):
    """
    Uses ffmpeg to produce a set of
    hls stream files which can be
    hosted while being generated.
    """

    self.state_cleanup()

    self.logger.info('Starting mp_queue_into_hls_stream')
    ff_in_args = {
      'nostats': None,
      'hide_banner':  None,
    }
    if self.debug:
      p_stderr = True
    else:
      p_stderr = False
      ff_in_args['loglevel'] = 'error' # type: ignore

    ff_in = ff.input('pipe:', **ff_in_args)
    ff_out_hls = ff.output(
      ff_in,
      (self.base_dir / 'stream.m3u8').absolute().as_posix(),
      hls_time=self.length,
      hls_list_size=0,
      hls_allow_cache=1,
      hls_segment_filename=(self.base_dir / 'stream_%06d.ts').absolute().as_posix(),
      hls_segment_type='mpegts',
      hls_flags='temp_file',
      hls_playlist_type='event'
    )
    ff_process = ff.run_async(ff_out_hls, pipe_stdin=True, pipe_stdout=False, pipe_stderr=p_stderr)
    if not ff_process.stdin:
      raise RuntimeError('Could not start hls ffmpeg process.')
    feed_thread, _ = u.mp_queue_to_pipe(
      self.stopper,
      self.logger,
      'passthrough_to_hls_out',
      ff_process.stdin,
      self.passthrough_queue
    )
    feed_thread.start()
    log_thread = None
    if self.debug:
      if not ff_process.stderr:
        self.logger.warning('Could not start logging for ffmpeg process, stderr was not exposed.')
      else:
        log_thread = u.ff_log_messages(self.stopper, self.logger, TextIOWrapper(ff_process.stderr))
        log_thread.start()

    return ff_process, feed_thread, log_thread

  def status_detail(self):
    if len(self.subprocesses) == 0:
      if len(self.threads) > 0:
        return 'Waiting/Blocked'
      return 'Stopped'
    if len(self.subprocesses) == 1:
      stdin = self.subprocesses[0].stdin
      if self.stopper.is_set():
        if self.subprocesses[0].poll() is None:
          return 'Stopping/Blocked'
        return 'Stopped'
      if stdin:
        pipe_size = os.fstat(stdin.fileno()).st_size
      else:
        pipe_size = 0
      return f'ff| [{pipe_size}], Vid Q [{self.passthrough_queue.qsize() * 8}]'
    return 'Unknown/Unexpected State'

  def start(self):
    """
    Starts the output subprocess
    along with all necessary threads.
    """
    ff_process, feed_thread, ff_log_thread = self.__mp_queue_into_hls_stream()
    self.subprocesses.append(ff_process)
    self.threads.append(feed_thread)
    if ff_log_thread:
      self.threads.append(ff_log_thread)

  def state_cleanup(self):
    # Delete all files in base_dir
    if len(list(self.base_dir.glob('*'))) > 0:
      self.logger.info('Deleting files in {dir}', dir=self.base_dir.absolute().as_posix())
      for file in self.base_dir.glob('*'):
        file.unlink()

def url_into_pipe(
    stop: Event,
    _logger: loguru.Logger,
    debug: bool,
    input_config: StreamInputConfig):
  """
  Uses yt-dlp and produce a stream of
  audiovideo data from a source url.
  """
  base_path = pathlib.Path(input_config['cache_dir'])
  yt_dlp_cache = pathlib.Path(input_config['cache_dir']) / 'yt-dlp-cache'
  if not base_path.exists():
    base_path.mkdir(parents=True, exist_ok=True)
  if not yt_dlp_cache.exists():
    yt_dlp_cache.mkdir(exist_ok=True)

  args = [
      'yt-dlp', input_config['url'],
      '--cache-dir', "./yt-dlp-cache",
      '-o', '-', 
  ]
  if input_config['quality'] != 'best':
    vid_format = f'bestvideo[height<={input_config["quality"]}]+bestaudio/best[height<={input_config["quality"]}]' \
                 if input_config['quality'].isnumeric() else input_config['quality']
    args.extend(['-f', vid_format])
  if input_config['start'] != "0":
    args.extend(["--download-sections", f'*{input_config["start"]}-inf'])
  yt_dlp_process = subprocess.Popen(
      args,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      cwd=input_config['cache_dir'],
      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
  )
  log_thread = None
  if _logger:
    if not yt_dlp_process.stderr:
      _logger.warning('Could not start logging for yt-dlp process, stderr was not exposed.')
    else:
      log_thread = u.ytdl_log_messages(stop, _logger, debug, TextIOWrapper(yt_dlp_process.stderr))
      log_thread.start()
  return yt_dlp_process, log_thread

def pipe_into_mp_queue(
    stop: Event,
    _logger: loguru.Logger,
    debug: bool,
    input_process: subprocess.Popen[bytes],
    pcm_output_queue: mp.Queue,
    passthrough_queue: mp.Queue):
  """
  Uses ffmpeg to produce a stream of
  pcm data, and also stream the input
  back to a pipe for display.
  """
  _logger.info('Starting pipe_into_mp_pipe')
  ff_in_args = {
    'nostats': None,
    'hide_banner':  None,
  }
  if debug:
    p_stderr = True
  else:
    p_stderr = False
    ff_in_args['loglevel'] = 'error' # type: ignore

  ff_in = ff.input('pipe:', **ff_in_args)
  ff_audio_map = ff_in.audio
  ff_out_pcm = ff.output(ff_audio_map, 'pipe:', acodec='pcm_f32le', ac=1, ar='16k', format='f32le')
  ff_process = ff.run_async(ff_out_pcm, pipe_stdin=True, pipe_stdout=True, pipe_stderr=p_stderr)
  if not input_process.stdout:
    raise RuntimeError('Could not start yt-dlp process')
  if not ff_process.stdout or not ff_process.stdin:
    raise RuntimeError('Could not start pcm ffmpeg process.')
  feed_thread = u.pipe_split(
    stop,
    _logger,
    'pipe_split_ytdl_ffmpeg_passthrough', 
    8192,
    input_process.stdout,
    ff_process.stdin,
    passthrough_queue
  )
  feed_thread.start()
  pcm_feed_thread = u.pipe_to_mp_queue(
    stop,
    _logger,
    'pcm_output_to_transx',
    1024,
    ff_process.stdout,
    pcm_output_queue
  )
  pcm_feed_thread.start()
  log_thread = None
  if debug:
    if not ff_process.stderr:
      _logger.warning('Could not start logging for ffmpeg process, stderr was not exposed.')
    else:
      log_thread = u.ff_log_messages(stop, _logger, TextIOWrapper(ff_process.stderr))
      log_thread.start()

  return ff_process, feed_thread, pcm_feed_thread, log_thread

def url_into_pcm_pipe(
    stop: Event,
    _logger: loguru.Logger,
    base_dir: str,
    url: str, 
    start: str):
  """
  Uses yt-dlp and ffmpeg to produce a stream of pcm data
  from a source url.
  """
  base_path = pathlib.Path(base_dir)
  yt_dlp_cache = pathlib.Path(base_dir) / 'yt-dlp-cache'
  if not base_path.exists():
    base_path.mkdir(parents=True, exist_ok=True)
  if not yt_dlp_cache.exists():
    yt_dlp_cache.mkdir(exist_ok=True)
  args = [
      'yt-dlp', url,
      '--cache-dir', "./yt-dlp-cache",
      '-S', '+size',
      '-o', '-', 
      '-f', 'ba*[acodec!*=aac]'
  ]
  if start != "0":
    args.extend(["--download-sections", f'*{start}-inf'])
  yt_dlp_process = subprocess.Popen(
      args,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,
      cwd=base_dir,
      creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
  )
  if not yt_dlp_process.stdout:
    raise RuntimeError('Could not start yt-dlp process for {url}')
  _logger.info('Starting pcm ffmpeg process.')
  ff_in_args = {
    'nostats': None,
    'hide_banner':  None,
    'loglevel': 'error'
  }
  ff_in = ff.input('pipe:', **ff_in_args)
  ff_audio_map = ff_in.audio
  ff_out_pcm = ff.output(ff_audio_map, 'pipe:', acodec='pcm_f32le', ac=1, ar='16k', format='f32le')
  ff_process = ff.run_async(ff_out_pcm, pipe_stdout=True, pipe_stdin=True)
  if not ff_process.stdout or not ff_process.stdin:
    raise RuntimeError('Could not start pcm ffmpeg process.')
  feed_thread = u.pipe_to_pipe(stop, _logger, 'ytdlp_to_ffmepg_pcm', 8192, yt_dlp_process.stdout, ff_process.stdin)
  feed_thread.start()
  return yt_dlp_process, ff_process, feed_thread

def probe(_logger: loguru.Logger, url: str, quality: str):
  """
  Uses yt-dlp to probe the source url for information
  """
  _logger.info('Starting yt-dlp process.')
  vid_format = f'bestvideo[height<={quality}]+bestaudio/best[height<={quality}]' \
               if quality.isnumeric() else quality
  args = [
      "yt-dlp", url,
      "-j",
      "-f", vid_format
  ]
  yt_dlp_process = subprocess.Popen(
      args,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL
  )
  out, _err = yt_dlp_process.communicate(None, None)

  return json.decoder.JSONDecoder().decode(out.decode('utf-8'))
