from __future__ import annotations
from multiprocessing.synchronize import Event
import loguru
import time
import queue
import multiprocessing as mp
from typing import Callable, Concatenate, Literal, ParamSpec, Tuple
from multiprocessing.connection import PipeConnection
from threading import Lock
from smatter.utils import get_logger

P = ParamSpec('P')
def guruize(fun: Callable[Concatenate[loguru.Logger, P], None]):
  def inner(*args: P.args, **kwargs: P.kwargs):
    fun(get_logger(), *args, **kwargs)
  return inner

@guruize
def mvp_log(_logger: loguru.Logger, level: str, prefix: str, text: str):
  l_map = {
    'TRACE': 5,
    'DEBUG': 10,
    'INFO': 20,
    'WARN': 30,
    'ERROR': 40,
    'CRITICAL': 50
  }
  _logger.log(l_map[level.upper()], f'{prefix}: {text}')


class TranslationDisplay:
  def __init__(self, queue: mp.Queue, ass_enable: bytes):
    self.ass_enable = ass_enable
    self.queue = queue
    self.waiting_transx: Tuple[float, float, str | None] | None = queue.get()
    self.last_end = 0.0
    self.finished = False

  def __ass_adjustment(self, text: str) -> bytes:
    ass_text = f'{{\\an2}}{text}'
    return b''.join((self.ass_enable, ass_text.encode()))
  
  def __gen_result(self, time: float):
    if self.waiting_transx and self.waiting_transx[0] <= time:
      if self.waiting_transx[2] != None:
        result = self.__ass_adjustment(self.waiting_transx[2]), str(int(max(1.0, (self.waiting_transx[1] - time) * 1000.0)))
      else:
        result = (None, None)
      self.waiting_transx = None
      return result
    else:
      return (None, None) 

  # Update Translation Shown
  def update_translation_display(self, time: float) -> Tuple[bytes | None, str | None, bool]:
    pause = False
    tbytes, t_time = self.__gen_result(time)
    if not self.waiting_transx and not self.finished:
      try:
        self.waiting_transx = self.queue.get_nowait()
        if self.waiting_transx == None:
          self.finished = True
        else:
          self.last_end = self.waiting_transx[1]
      except queue.Empty as e:
        if self.last_end <= time:
          pause = True
    if not self.finished:
      return tbytes, t_time, pause
    else:
      return None, None, False

def show_mpv_transx_window(
    stop: Event,
    _logger: loguru.Logger,
    transx_out: mp.Queue,
    passthrough_queue: mp.Queue, 
    thumb_url: str,
    name: str,
    status_fun: Callable[[], bool]
    ):
  
  import mpv #type: ignore
  kwargs = {
    'log_handler': mvp_log,
    'input_default_bindings': True,
    'input_vo_keyboard': True, 
    'osc': True
  }
  _logger.info('Starting player')
  player = mpv.MPV(**kwargs)
  
  _logger.info('Registering passthrough stream')
  @player.python_stream('piped_stream')
  def pipe_reader():
    try:
      while b:= passthrough_queue.get():
        if b == None:
          yield b''
          break
        else:
          yield b
    except Exception as e:
      _logger.exception(e)
      yield b''

  try:
    status_fun_ok = True
    offset = 0.0
    ass_enable: bytes = player.command('expand-text', '${osd-ass-cc/0}', decoder=mpv.identity_decoder) #type: ignore
    player._set_property('image-display-duration', 'inf')
    player._set_property('title', name)
    player.play(thumb_url)
    _logger.warning('Waiting for transx to show up before playing...')
    while transx_out.empty():
      if status_fun_ok:
        status_fun_ok = status_fun()
      if player.core_shutdown:
        raise Exception('Player is shutting down before load completed')
      time.sleep(1)
    _logger.info('Play is now starting...')
    display = TranslationDisplay(transx_out, ass_enable)
    player.play('python://piped_stream')
    offset_lock = Lock() 

    _logger.info('Registering bindings, observer, and title')
    player.unregister_key_binding('Shift+Ctrl+LEFT')
    player.unregister_key_binding('Shift+Ctrl+RIGHT')

    @player.key_binding('Shift+Ctrl+LEFT')
    def sub_shift_left(state: Literal['u-','d-'], _name, _char):
      nonlocal offset, offset_lock
      if state == 'd-':
        with offset_lock:
          offset -= 0.5
        player._set_property('title', f'{name} (sub offset: {round(offset, 2)})')
        _logger.info('Adjusting subtitle delay backwards, offset is now {os}', os=offset)
    
    @player.key_binding('Shift+Ctrl+RIGHT')
    def sub_shift_right(state: Literal['u-','d-'], _name, _char):
      nonlocal offset, offset_lock
      if state == 'd-':
        with offset_lock:
          offset += 0.5
        player._set_property('title', f'{name} (sub offset: {round(offset, 2)})')
        _logger.info('Adjusting subtitle delay backwards, offset is now {os}', os=offset)
    
    @player.property_observer('time-pos')
    def time_observer(_name, value):
      nonlocal offset, offset_lock
      # Here, _value is either None if nothing is playing or a float containing
      # fractional seconds since the beginning of the file.
      if (value != None):
        try:
          if not display.finished:
            with offset_lock:
              value -= offset
            tbytes, d_time, pause = display.update_translation_display(value)
            warn_count = 0
            if pause:
              _logger.warning('Transx is taking too long and can\'t keep up with playback! Pausing video.')
            while pause and not display.finished:
              player._set_property('pause', True)
              time.sleep(0.1)
              warn_count += 1
              if not warn_count % 50:
                _logger.warning('Waited {n} seconds for new translations...', n=(warn_count / 10))
              tbytes, d_time, pause = display.update_translation_display(value)
            else:
              if player._get_property('pause'):
                player._set_property('pause', False)
            if tbytes:
              player.command_async('show-text', tbytes, d_time)
        except Exception as e:
          _logger.exception(e)

    player._set_property('title', f'{name} (sub offset: {round(offset, 2)})')
    _logger.info('MPV load complete, now playing.')
    
    while True:
      if status_fun_ok:
        status_fun_ok = status_fun()
      if player.core_shutdown:
        break
      time.sleep(0.2)
  except Exception as e:
    _logger.exception(e)
  finally:
    stop.set()