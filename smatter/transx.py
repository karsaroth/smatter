from __future__ import annotations
from multiprocessing.connection import PipeConnection
import re, time, string
from subprocess import Popen
import numpy as np
import numpy.typing as npt
import multiprocessing as mp
import threading as th
from typing import IO, Callable, Generator, Literal, TypedDict, Tuple, List, Dict, Any
from multiprocessing.synchronize import Event
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from datetime import datetime, timedelta
import loguru
from loguru import logger
from smatter.ff_process import url_into_pcm_pipe
from libs.vad.utils_vad import VADIterator
import smatter.utils as u

class TransXConfig(TypedDict):
  stop: Event
  output_queue: mp.Queue
  _logger: loguru.Logger
  base_path: str
  stream_url: str
  format: Literal['srt', 'vtt', 'plain']
  requested_start: str

class WhisperConfig(TypedDict):
  model_size: str
  force_gpu: bool
  lang: str
  goal: str

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

def fix_repeated_phrases(string: str) -> Tuple[str, bool]:
  """
  Whisper likes to return segments with 
  phrases repeated many times sometimes
  This is designed to make that more readable.    
  """
  pattern = r'((\b\w+\b\W+)+?)(=?\1{3,})'
  parts_iter = re.finditer(pattern, string)
  
  final = ''
  end_pos = 0
  for part in parts_iter:
    final += string[:part.start()]
    final += part.group(1)
    end_pos = part.end()
  # No issues found
  if final == '':
    return (string, False)
  else:
    final += string[end_pos:] + '[...]'
    return (final, True)

def fix_repeated_sounds(string: str) -> Tuple[str, bool]:
  """
  Whisper likes to turn a chuckle into
  a maniacal laugh with hundreds of characters.
  This is designed to make that more readable.  
  """
  pattern = r'((\w+)(?=\2{5}))+?'
  parts = re.findall(pattern, string)

  if len(parts) == 0:
    return (string, False)
  smallest, _ = parts.pop()
  final = ''
  current = ''
  count = 0
  for c in string:
    current += c
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
  if len(final) < len(string):
    final += '[...]'
  return (final, True)

def filter_gigo_results(
    transxs: List[TransXData], 
    gigo_phrase_list: List[str]
  ) -> List[TransXData]:
  """
  A last ditch attempt to remove 
  garbage results from music, noise,
  or silence being fed into the model.
  """
  return list(filter(lambda x: 
                        x["text"]
                        .translate(str.maketrans('', '', string.punctuation))
                        .casefold() 
                        not in gigo_phrase_list, 
                      transxs)
          )

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
  text, was_fixed = fix_repeated_phrases(transx["text"])
  if not was_fixed:
    text, was_fixed = fix_repeated_sounds(text)
  c = '!' if (transx["probability"] < 0.3) else '?' if (transx["probability"] < 0.5) else '-'
  n = '!' if (transx["noise_probability"] > 0.7) else '?' if (transx["noise_probability"] > 0.5) else '-'
  r = '!' if (transx["compression_ratio"] > 3.0) else '?' if (transx["compression_ratio"] > 2.0) else '-'
  return f'[{c}{n}{r}]: {text}'

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

def join_similar(transx: List[TransXData]):
  """
  Sometimes Whisper returns the same segment result multiple times, or the same
  audio part translated slightly differently. This removes duplicates
  and picks the better version of a duplicate
  """
  if len(transx) > 1:
    for i in range(len(transx) - 1):
      if transx[i]['start'] == transx[i + 1]['start'] and \
          transx[i]['end'] == transx[i + 1]['end'] and \
          transx[i]['text'] == transx[i + 1]['text']:
        transx.pop(i + 1)
        return join_similar(transx)
      elif transx[i]['text'] == transx[i + 1]['text']:
        joiner1 = transx[i]
        joiner2 = transx[i + 1]
        transx.insert(i, {
          'start': min(joiner1['start'], joiner2['start']),
          'end': max(joiner1['end'], joiner2['end']),
          'probability': max(joiner1['probability'], joiner2['probability']),
          'noise_probability': max(joiner1['noise_probability'], joiner2['noise_probability']),
          'compression_ratio': max(joiner1['compression_ratio'], joiner2['compression_ratio']),
          'text': joiner1['text']
        })
        transx.remove(joiner1)
        transx.remove(joiner2)
        return join_similar(transx)
  return transx

def chunk_from_samples(
    _stop: Event, 
    _logger: loguru.Logger,
    r_fun: Callable[[int], bytes],
    length: int
  ) -> Generator[npt.NDArray[np.float32], None, None]:
  """
  Generator of float32 chunks from
  the bytes coming out of stdout
  from the provided process
  """

  #Prime the generator
  _buffer = r_fun(length)
  def _gen(stop: Event, buffer: bytes):
    try:
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
            raise Exception('Chunk buffer overflow, calculations off.')
        buffer = r_fun(length)
    except Exception as e:
      _logger.exception(e)
  
  return _gen(_stop, _buffer)

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
  vad.current_sample = (start * 16000)
  for chunk in chunks:
    if len(chunk) != chunk_size:
      raise Exception('Chunk size incorrect, something went wrong with chunk creation')
    is_active = 'start' in active_voice.keys()
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
      if 'start' in result.keys():
        silence_start = -1
        active_voice['start'] = int(result['start'])
        voice_samples[watermark:watermark + chunk_size] = prev_chunk
        watermark += chunk_size
        voice_samples[watermark:watermark + chunk_size] = chunk
        watermark += chunk_size
        vs_offset = sample_offset
        _logger.info('Voice activity start at {}', seconds_to_timestamp(float(vs_offset) / 16000, True))
      if 'end' in result.keys() and is_active:
        active_voice['end'] = int(result['end'])
        _logger.info('Voice activity end at {}', seconds_to_timestamp(float(active_voice['end']) / 16000, True))
    if is_active and watermark + chunk_size > max_size:
      active_voice['end'] = vad.current_sample
      vad.reset_states()
      vad.current_sample = active_voice['end']
      _logger.warning('Voice activity cut off at {}, may result in strange translation', seconds_to_timestamp((float(watermark) / 16000)))
    if is_active and 'end' in active_voice.keys():
      _logger.debug('Voice activity buffer ready for TransX. Length: {}', float(active_voice['end'] - active_voice['start']) / 16000)
      # Voice goes for processing
      yield active_voice['start'], np.copy(voice_samples[active_voice['start'] - vs_offset:active_voice['end'] - vs_offset])
      watermark = 0
      active_voice = {}
    prev_chunk = chunk

def transx_from_audio_stream(
    transx_config: TransXConfig,
    whisper_config: WhisperConfig,
    sync_pipe: PipeConnection | None,
    ):
  """
  Get a stdout stream from ffmpeg based on the provided
  url and generated translations from it as fast as
  possible.
  """
  _logger = transx_config['_logger']
  close_fast = False
  p_list: List[Popen[bytes]] = []
  try:
    gigo_phrases, model = transx_prep(_logger, whisper_config)
    ytdl_process, ff_process, _feed_thread = url_into_pcm_pipe(
      transx_config['stop'], 
      transx_config['_logger'],
      transx_config['base_path'],
      transx_config['stream_url'],
      transx_config['requested_start'],
      sync_pipe
    )
    if ff_process.stdout is None:
      raise Exception('Couldn\'t read from ffmpeg')
    p_list.append(ytdl_process)
    p_list.append(ff_process)
    if not sync_pipe:
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
    else:
      _logger.debug('Using broadcast start time for live stream to sync.')
      start_int = 0
    chunk_gen = chunk_from_samples(transx_config['stop'], _logger, ff_process.stdout.read, 1024)
    # Should be second sync timestamp
    if sync_pipe:
      sync_pipe.send(time.time())
    run_transx(transx_config, whisper_config, start_int, model, gigo_phrases, chunk_gen)
  except Exception as e:
    _logger.exception(e)
    close_fast = True
  finally:
    _logger.debug('Transx process is finishing')
    u.close_queue(transx_config['output_queue'], transx_config['stop'], _logger, close_fast)
    _logger.info('Transx process finished')
    for p in p_list:
      if p.poll() is None:
        _logger.debug('Transx process is finishing')
        p.terminate()
        _logger.debug('Transx process finished')

def transx_from_queue(
    transx_config: TransXConfig,
    whisper_config: WhisperConfig,
    input_queue: mp.Queue,
  ):
  """
  Get a stdout stream from the input pipe
  and generated translations from it as fast as
  possible.
  """
  _logger = transx_config['_logger']
  stop = transx_config['stop']
  transx_output_queue = transx_config['output_queue']
  close_fast = False
  try:
    gigo, model = transx_prep(_logger, whisper_config)
    _logger.debug('Setting up start-point for translation to match requested start time.')
    # Assuming destination will use start-from-zero for now.
    start_int = 0
    def r_fun(_: int):
      return input_queue.get()
    chunk_gen = chunk_from_samples(stop, _logger, r_fun, 1024)
    run_transx(transx_config, whisper_config, start_int, model, gigo, chunk_gen)
  except Exception as e:
    _logger.exception(e)
    close_fast = True
  finally:
    _logger.debug('Transx process is finishing')
    u.close_queue(transx_output_queue, stop, _logger, close_fast)
    _logger.info('Transx process finished')

def transx_prep(
    _logger: loguru.Logger,
    whisper_config: WhisperConfig
    ):
  gigo_phrases = []
  with open('gigo_phrases.txt', 'r') as f:
    gigo_phrases = list(map(lambda x: x.strip().translate(str.maketrans('', '', string.punctuation)).casefold(), f.read().splitlines()))
  _logger.info('Loading Whisper model')
  model = WhisperModel(
    whisper_config['model_size'],
    device= "cuda" if whisper_config['force_gpu'] else "auto",
    compute_type="default"
  )
  _logger.info('Model loading complete')
  return gigo_phrases, model

def run_transx(
    transx_config: TransXConfig,
    whisper_config: WhisperConfig,
    start_int: int,
    model: WhisperModel,
    gigo_phrases: List[str],
    chunk_gen: Generator[npt.NDArray[np.float32], None, None]
    ):
  _logger = transx_config['_logger']
  _format = transx_config['format']
  stop = transx_config['stop']
  transx_output_queue = transx_config['output_queue']
  transx_kwargs = {
    'language': whisper_config['lang'], 
    'task': whisper_config['goal'], 
    'condition_on_previous_text': False
  }
  count = 0
  blanks_queue = transx_output_queue if _format == 'plain' else None
  for start, voice in vad_samples(_logger, chunk_gen, 1024, 320000, start_int, blanks_queue):
    if stop.is_set():
      break
    start_time = (float(start) / 16000.0)
    _logger.debug(f'Starting speech transx for {start_time}')
    segments, _ = model.transcribe(audio=voice, **transx_kwargs)
    _logger.debug(f'Finished speech transx for {start_time}')
    transx_segments = list(map(lambda s: segment_to_txdata(s, start_time), segments))
    _logger.debug(f'Whisper found {len(transx_segments)} transx results')
    ready_segments = join_similar(transx_segments)
    final_segments = filter_gigo_results(ready_segments, gigo_phrases)
    _logger.debug(f'Cleaned down to {len(final_segments)} transx results.')
    for t in final_segments:
      count += 1
      if _format == 'plain':
        transx_output_queue.put((t['start'], t['end'], transx_to_string(t)))
      else:
        transx_output_queue.put((t['start'], t['end'], txdata_to_srt(t, count, _format == 'vtt')))
    _logger.info(f'TransX is now at {count} translations')
