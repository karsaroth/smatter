"""
Translation and Transcription (Transx)
functionality for smatter
"""

from __future__ import annotations
import queue
import re
import string
import abc
import multiprocessing as mp
from subprocess import Popen
from multiprocessing.synchronize import Event
from multiprocessing.connection import PipeConnection
from typing import Callable, Generator, Literal, TypedDict, Tuple, List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import numpy.typing as npt
from faster_whisper import WhisperModel, download_model
from faster_whisper.transcribe import Segment
import loguru
from loguru import logger
from smatter.ff_process import url_into_pcm_pipe
from libs.vad.utils_vad import VADIterator
import smatter.utils as u

class TransXConfig(TypedDict):
  """
  Configuration for the TransX
  process
  """
  stop: Event
  output_queue: mp.Queue
  _logger: loguru.Logger
  base_path: str
  stream_url: str
  format: Literal['srt', 'vtt', 'plain']
  requested_start: str
  model_config: TypedDict

class WhisperConfig(TypedDict):
  """
  Configuration for faster whisper
  when used with smatter
  """
  model: Literal['faster_whisper']
  model_size: str
  force_gpu: bool
  lang: str
  goal: str
  model_root: str
  gigo_phrases: List[str]

class TransXData(TypedDict):
  """
  Designed to hold useful
  Transx data for use in
  preparing translations
  """
  start: float
  end: float
  probability: float
  noise_probability: float
  compression_ratio: float
  text: str

class TransXModel(abc.ABC):
  """
  Wrap model handling so that Whisper
  could be replaced more easily if
  wanted in the future
  """

  @abc.abstractmethod
  def prepare_dependencies(self, local=False) -> str:
    """
    Prepare any dependencies, return
    a path to the model or other 
    indication of what has been prepared.
    """

  @abc.abstractmethod
  def prepare_model(self):
    """
    Load the model into memory
    ready to be used in the
    transx method
    """

  @abc.abstractmethod
  def transx(
    self,
    audio: np.ndarray,
    start_offset: float,
    language: Optional[str],
    goal: Optional[Literal['transcribe', 'translate']],
    **kwargs) -> List[TransXData]:
    """
    Generate transx of the provided
    audio. Timestamps must be offset
    by the specified start_offset.
    """

class WhisperTransXModel(TransXModel):
  """
  Whisper model handling
  """
  REPEAT_PHRASE_PATTERN = r'((\b\w+\b\W+)+?)(=?\1{3,})'
  REPEAT_SOUND_PATTERN = r'((\w+)(?=\2{5}))+?'

  def __init__(self, _logger: loguru.Logger, config: WhisperConfig):
    self.logger = _logger
    self.config = config
    self.model: WhisperModel | None = None

  @staticmethod
  def join_similar(transx: List[TransXData]):
    """
    Sometimes Whisper returns the same segment result multiple times, or the same
    audio part translated slightly differently. This removes duplicates
    and picks the better version of a duplicate
    """
    i = 0
    while i < len(transx) - 1:
      if transx[i]['start'] == transx[i + 1]['start'] and \
            transx[i]['end'] == transx[i + 1]['end'] and \
            transx[i]['text'] == transx[i + 1]['text']:
        transx.pop(i + 1)
      elif transx[i]['text'] == transx[i + 1]['text']:
        joiner1 = transx[i]
        joiner2 = transx[i + 1]
        transx[i] = TransXData(
          start = min(joiner1['start'], joiner2['start']),
          end = max(joiner1['end'], joiner2['end']),
          probability = max(joiner1['probability'], joiner2['probability']),
          noise_probability = max(joiner1['noise_probability'], joiner2['noise_probability']),
          compression_ratio = max(joiner1['compression_ratio'], joiner2['compression_ratio']),
          text = joiner1['text']
        )
        transx.pop(i + 1)
      else:
        i += 1
    return transx

  def __filter_gigo_results(self, transxs: List[TransXData]) -> List[TransXData]:
    """
    A last ditch attempt to remove 
    garbage results from music, noise,
    or silence being fed into the model.
    """
    return list(
      filter(
        lambda x:
          x["text"]
            .translate(str.maketrans('', '', string.punctuation))
            .casefold()
            not in self.config['gigo_phrases'],
          transxs
      )
    )

  @staticmethod
  def fix_repeated_phrases(transx: TransXData) -> bool:
    """
    Whisper likes to return segments with 
    phrases repeated many times sometimes
    This is designed to make that more readable.    
    """
    pattern = WhisperTransXModel.REPEAT_PHRASE_PATTERN
    text = transx['text']
    parts_iter = re.finditer(pattern, text)

    final = ''
    end_pos = 0
    for part in parts_iter:
      final += text[:part.start()]
      final += part.group(1)
      end_pos = part.end()
    # No issues found
    if final == '':
      return False
    else:
      final += text[end_pos:] + '[...]'
      transx['text'] = final
      return True

  @staticmethod
  def fix_repeated_sounds(transx: TransXData) -> bool:
    """
    Whisper likes to turn a chuckle into
    a maniacal laugh with hundreds of characters.
    This is designed to make that more readable.  
    """
    pattern = WhisperTransXModel.REPEAT_SOUND_PATTERN
    text = transx['text']
    parts = re.findall(pattern, text)

    if len(parts) == 0:
      return False
    smallest, _ = parts.pop()
    final = ''
    current = ''
    count = 0
    for character in text:
      current += character
      if len(current) >= len(smallest):
        if current == smallest:
          if count > 5:
            current = ''
          else:
            count += 1
            final += current
            current = ''
        else:
          final += current[0]
          current = current[1:]
    if current != '':
      final += current
    if len(final) < len(text):
      final += '[...]'
    transx['text'] = final
    return True

  @staticmethod
  def segment_to_txdata(segment: Segment, segment_start_time: float) -> TransXData:
    """
    Prep TX data for analysis
    """
    return {
      'start': segment_start_time + segment.start,
      'end': segment_start_time + segment.end,
      'probability': np.exp(segment.avg_logprob),
      'noise_probability': segment.no_speech_prob,
      'compression_ratio': segment.compression_ratio,
      'text': segment.text.strip()
    }

  def prepare_dependencies(self, local=False):
    return download_model(
      self.config['model_size'],
      local_files_only=local,
      cache_dir=self.config['model_root'],
    )

  def prepare_model(self):
    self.model = WhisperModel(
      self.config['model_size'],
      device = "cuda" if self.config['force_gpu'] else "auto",
      compute_type = "default",
      download_root = self.config['model_root']
    )

  def transx(
      self,
      audio: np.ndarray,
      start_offset: float,
      language: Optional[str],
      goal: Optional[Literal['transcribe', 'translate']],
      **kwargs) -> List[TransXData]:
    if self.model is None:
      raise RuntimeError('Model not loaded')
    transx_kwargs = {
      'language': language if language else self.config['lang'], 
      'task': goal if goal else self.config['goal'], 
      'condition_on_previous_text': kwargs['context'] if 'context' in kwargs else False
    }
    self.logger.debug(f'Starting speech transx for {start_offset}')
    segments, _ = self.model.transcribe(audio=audio, **transx_kwargs)
    self.logger.debug(f'Finished speech transx for {start_offset}')
    transx_list = list(
      map(lambda s: WhisperTransXModel.segment_to_txdata(s, start_offset), segments)
    )
    self.logger.debug(f'Whisper found {len(transx_list)} transx results')

    # Filter for unnecessary or noisy transx
    final_list = self.__filter_gigo_results(
      WhisperTransXModel.join_similar(
        transx_list
      )
    )

    # Improve readability of some transx
    for transx in final_list:
      was_fixed = WhisperTransXModel.fix_repeated_phrases(transx)
      if not was_fixed:
        WhisperTransXModel.fix_repeated_sounds(transx)

    self.logger.debug(f'Cleaned and filtered down to {len(final_list)} transx results.')
    return final_list

class InteractiveTransXProcess():
  """
  A class to handle the process of
  translating a stream into subtitles
  """
  def __init__(
      self,
      _logger: loguru.Logger,
      model_config: TypedDict,
      ):
    self.stopper = mp.Event()
    self.input_queue = mp.Queue()
    self.output_queue = mp.Queue()
    self.logger = _logger
    self.model_config = model_config
    self.format: Literal['srt', 'vtt', 'plain'] = 'plain'
    self.__process: Optional[mp.Process] = None

  def check_model(self, download=False):
    """
    Check the TransX model
    is available and ready
    """
    verify_and_prepare_models(self.logger, self.model_config, download)

  def status(self):
    """
    Return the status of the
    TransX process
    """
    if self.__process is None:
      return False
    if self.__process.is_alive():
      return True
    return False

  def status_detail(self):
    """
    Return the status of the
    TransX process in detail
    """
    if self.__process is None:
      return 'Stopped'
    if self.__process.is_alive() and not self.stopper.is_set():
      return 'Running (extra info TODO)'
    if self.__process.is_alive() and self.stopper.is_set():
      return 'Stopping/Blocked'
    return 'Stopped/Complete'

  def start(
      self,
      requested_start: str,
      language: Optional[str],
      goal: Optional[Literal['transcribe', 'translate']]):
    """
    Start the TransX process
    based on the current
    configuration
    """
    self.stopper.clear()
    tx_config = TransXConfig(
      _logger = self.logger,
      stop = self.stopper,
      output_queue=self.output_queue,
      model_config=self.model_config,
      format=self.format,
      requested_start=requested_start,
      base_path='',
      stream_url='',
    )
    tx_piped_args: Tuple[
        TransXConfig,
        mp.Queue,
        Optional[str],
        Optional[Literal['transcribe', 'translate']]] = (
      tx_config, self.input_queue, language, goal
    )
    self.__process = mp.Process(target=transx_from_queue, args=tx_piped_args)
    self.__process.start()

  def stop(self):
    """
    Stops the TransX process
    and cleans up the process
    """
    self.stopper.set()

  def cleanup(self):
    """
    Clean up processes and
    state
    """
    if self.__process is not None:
      self.logger.info('Waiting for TransX process to finish')
      self.__process.join()
      self.__process = None
      self.logger.info('TransX process finished')
    while not self.input_queue.empty():
      try:
        # Clean out the output queue
        self.input_queue.get_nowait()
      except queue.Empty:
        break
    self.logger.info('Transx cleanup complete')

def seconds_to_timestamp(seconds, vtt = False):
  """
  Format the timestamp as "HH:mm:ss,SSS" (or "HH:mm:ss.SSS" for VTT)
  """
  timedelta_obj = timedelta(seconds=seconds)
  start_point = datetime(1970, 1, 1)
  timestamp = start_point + timedelta_obj
  # Exclude last 3 digits for milliseconds
  strf_t = f"%H:%M:%S{'.' if vtt else ','}%f"
  return timestamp.strftime(strf_t)[:-3]

def transx_to_string(transx: TransXData) -> str:
  """
  Takes translation info (probablity etc) and appends
  indicators to the beginning of the translation. ? for
  uncertain translations and ! for very uncertain.
  """
  confidence = '!' if (transx["probability"] < 0.3) else \
               '?' if (transx["probability"] < 0.5) else '-'
  noise = '!' if (transx["noise_probability"] > 0.7) else \
          '?' if (transx["noise_probability"] > 0.5) else '-'
  compression_ratio = '!' if (transx["compression_ratio"] > 3.0) else \
                      '?' if (transx["compression_ratio"] > 2.0) else '-'

  return f'[{confidence}{noise}{compression_ratio}]: {transx["text"]}'

def txdata_to_srt(transx: TransXData, num: int, vtt = False):
  """
  Converts The TransXData into an SRT subtitle (looking like this):\n
  <num>\n
  XX:XX:XX,XXX --> XX:XX:XX,XXX\n
  <text>\n
  <blank line>
  """
  start = seconds_to_timestamp(transx['start'], vtt)
  end = seconds_to_timestamp(transx['end'], vtt)
  return f'{num}\n{start} --> {end}\n{transx_to_string(transx)}\n\n'

def chunk_from_samples(
    _stop: Event,
    r_fun: Callable[[int], bytes | None],
    length: int
  ) -> Generator[npt.NDArray[np.float32], None, None]:
  """
  Generator of float32 chunks from
  the bytes coming out of stdout
  from the provided process
  """

  #Prime the generator
  _buffer = r_fun(length)
  def gen(stop: Event, buffer: bytes | None):
    chunk = np.zeros(length, np.float32)
    chunk_watermark = 0
    while not stop.is_set() and buffer:
      np_buffer = np.frombuffer(buffer, np.float32)
      np_buffer_length = len(np_buffer)
      samples_taken = None
      samples_remaining = None
      if chunk_watermark == 0 and np_buffer_length == length:
        yield np_buffer
      else:
        # Dealing with smaller than requested buffer sizes
        if chunk_watermark + np_buffer_length > length:
          samples_taken = length - chunk_watermark
          samples_remaining = np_buffer_length - samples_taken
          chunk[chunk_watermark:] = np_buffer[:samples_taken]
          chunk_watermark = length
        else:
          chunk[chunk_watermark:chunk_watermark + np_buffer_length] = np_buffer
          chunk_watermark += np_buffer_length
        if chunk_watermark == length:
          yield chunk
          chunk_watermark = 0
          if samples_taken and samples_remaining:
            chunk[chunk_watermark:chunk_watermark + samples_remaining] = np_buffer[samples_taken:]
            chunk_watermark += samples_remaining
        elif chunk_watermark > length:
          raise RuntimeError('Chunk buffer overflow, calculations off.')
      buffer = r_fun(length)

  return gen(_stop, _buffer)

def vad_samples(
  _logger: loguru.Logger,
  chunks: Generator[npt.NDArray[np.float32], Any, None],
  chunk_size: int,
  max_size: int,
  start: int,
  blanks: mp.Queue | None
  ) -> Generator[Tuple[int, np.ndarray[Any, np.dtype[np.float32]]], None, None]:
  """
    Skip samples until VAD activates, then cache them
    (up to max_size) until VAD deactivates,
    before returning the cache.
  """
  _logger.info('Loading VAD Model')
  vad = VADIterator()
  voice_samples = np.zeros(max_size, np.float32)
  prev_chunk = np.zeros(chunk_size, np.float32)
  vs_offset = 0
  watermark = 0
  silence_start = -1
  active_voice: Dict[str, int] = {}
  vad.current_sample = start * 16000
  for chunk in chunks:
    if len(chunk) != chunk_size:
      raise RuntimeError('Chunk size incorrect, something went wrong with chunk creation')
    is_active = 'start' in active_voice
    # Previous sample (before vad check of current sample)
    sample_offset = vad.current_sample
    if is_active:
      voice_samples[watermark:watermark + chunk_size] = chunk
      watermark += chunk_size
    elif blanks:
      # Long periods of silence could cause buffering on a livestream
      # due to the subtitle listener being unsure if the subtitle
      # generator is keeping up.
      if silence_start == -1:
        silence_start = sample_offset
      if sample_offset - silence_start > 16000:
        blanks.put((float(silence_start) / 16000.0, float(sample_offset) / 16000.0, None))
        silence_start = -1
    if result := vad(chunk, False):
      if 'start' in result:
        silence_start = -1
        active_voice['start'] = int(result['start'])
        voice_samples[watermark:watermark + chunk_size] = prev_chunk
        watermark += chunk_size
        voice_samples[watermark:watermark + chunk_size] = chunk
        watermark += chunk_size
        vs_offset = sample_offset
        _logger.info(
          'Voice activity start at {}', 
          seconds_to_timestamp(float(vs_offset) / 16000, True)
        )
      if 'end' in result and is_active:
        active_voice['end'] = int(result['end'])
        _logger.info(
          'Voice activity end at {}', 
          seconds_to_timestamp(float(active_voice['end']) / 16000, True)
        )
    if is_active and watermark + chunk_size > max_size:
      active_voice['end'] = vad.current_sample
      vad.reset_states()
      vad.current_sample = active_voice['end']
      _logger.warning(
        'Voice activity cut off at {}, may result in strange translation',
        seconds_to_timestamp((float(watermark) / 16000))
      )
    if is_active and 'end' in active_voice:
      _logger.debug(
        'Voice activity buffer ready for TransX. Length: {}',
         float(active_voice['end'] - active_voice['start']) / 16000
      )
      # Voice goes for processing
      yield active_voice['start'], np.copy(
        voice_samples[active_voice['start'] - vs_offset:active_voice['end'] - vs_offset]
      )
      watermark = 0
      active_voice = {}
    prev_chunk = chunk

def config_to_model(_logger: loguru.Logger, config: TypedDict) -> TransXModel:
  """
  Convert the config into a model
  """
  if config['model'] == 'faster_whisper':
    return WhisperTransXModel(_logger, config) # type: ignore
  raise RuntimeError('Unsupported model')

def verify_and_prepare_models(_logger: loguru.Logger, config: TypedDict, download=False):
  """
  Check if the model is downloaded and download it if not
  """
  model = config_to_model(_logger, config)
  model.prepare_dependencies(download)

def transx_from_audio_stream(transx_config: TransXConfig):
  """
  Get a stdout stream from ffmpeg based on the provided
  url and generated translations from it as fast as
  possible.
  """
  _logger = transx_config['_logger']
  close_fast = False
  process_list: List[Popen[bytes]] = []
  try:
    model = config_to_model(_logger, transx_config['model_config'])
    _logger.info('Loading model')
    model.prepare_model()
    _logger.info('Model loading complete')
    ytdl_process, ff_process, _feed_thread = url_into_pcm_pipe(
      transx_config['stop'],
      transx_config['_logger'],
      transx_config['base_path'],
      transx_config['stream_url'],
      transx_config['requested_start']
    )
    if ff_process.stdout is None:
      raise RuntimeError('Couldn\'t read from ffmpeg')
    process_list.append(ytdl_process)
    process_list.append(ff_process)

    _logger.debug('Setting up start-point for translation to match requested start time.')
    start_match = u.hms_match(transx_config['requested_start'])
    if start_match is None:
      start_int = 0
    else:
      start_int = int(
        timedelta(
          hours=int(start_match.group(1)) if start_match.group(1) else 0,
          minutes=int(start_match.group(2)) if start_match.group(2) else 0,
          seconds=int(start_match.group(3))
        )
        .total_seconds()
      )
    chunk_gen = chunk_from_samples(transx_config['stop'], ff_process.stdout.read, 1024)
    run_transx(transx_config, start_int, model, chunk_gen, None, None)
  except Exception as ex:
    _logger.exception(ex)
    close_fast = True
  finally:
    _logger.debug('Transx process is finishing')
    u.close_queue(transx_config['output_queue'], transx_config['stop'], _logger, close_fast)
    _logger.info('Transx process finished')
    for process in process_list:
      if process.poll() is None:
        _logger.debug('Transx process is finishing')
        process.terminate()
        _logger.debug('Transx process finished')

def transx_from_queue(
    transx_config: TransXConfig, 
    input_queue: mp.Queue,
    language: Optional[str],
    goal: Optional[Literal['transcribe', 'translate']]
    ):
  """
  Get a stdout stream from the input queue
  and generated translations from it as fast as
  possible.
  """
  _logger = transx_config['_logger']
  stop = transx_config['stop']
  transx_output_queue = transx_config['output_queue']
  close_fast = False
  try:
    model = config_to_model(_logger, transx_config['model_config'])
    _logger.info('Loading model')
    model.prepare_model()
    _logger.info('Model loading complete')
    _logger.debug('Setting up start-point for translation to match requested start time.')
    # Assuming destination will use start-from-zero for now.
    start_int = 0

    def r_fun(_: int):
      """
      Emulate a read operation
      on a pipe using the input
      queue. If stop gets set
      then shortcut to a None
      result.
      """
      while not stop.is_set():
        try:
          return input_queue.get(True, 1)
        except queue.Empty:
          continue
      return None
    chunk_gen = chunk_from_samples(stop, r_fun, 1024)
    run_transx(transx_config, start_int, model, chunk_gen, language, goal)
  except (KeyboardInterrupt, SystemExit):
    _logger.info('Transx process asked to exit, cleaning up.')
    close_fast = True
  except Exception as ex:
    _logger.exception(ex)
    close_fast = True
  finally:
    _logger.debug('Transx process is finishing')
    u.close_queue(transx_output_queue, stop, _logger, close_fast)
    _logger.info('Transx process finished')

def run_transx(
    transx_config: TransXConfig,
    start_int: int,
    model: TransXModel,
    chunk_gen: Generator[npt.NDArray[np.float32], None, None],
    language: Optional[str],
    goal: Optional[Literal['transcribe', 'translate']]
    ):
  """
  Will run transx on the provided
  chunks until the generator is
  exhausted.
  """
  _logger = transx_config['_logger']
  _format = transx_config['format']
  stop = transx_config['stop']
  transx_output_queue = transx_config['output_queue']
  count = 0
  blanks_queue = transx_output_queue if _format == 'plain' else None
  for start, voice in vad_samples(_logger, chunk_gen, 1024, 320000, start_int, blanks_queue):
    if stop.is_set():
      break
    start_time = float(start) / 16000.0
    transx_data = model.transx(voice, start_time, language, goal)
    for transx in transx_data:
      count += 1
      if _format == 'plain':
        transx_output_queue.put((transx['start'], transx['end'], transx_to_string(transx)))
      else:
        transx_output_queue.put((
          transx['start'], transx['end'],
          txdata_to_srt(transx, count, _format == 'vtt')
        ))
    _logger.info(f'TransX is now at {count} translations')
