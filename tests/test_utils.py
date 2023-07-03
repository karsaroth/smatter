from __future__ import annotations
from datetime import timedelta
import datetime, multiprocessing as mp, threading as th
from io import BytesIO
import time
from smatter import utils
import loguru
from loguru import logger
import pytest

@pytest.mark.parametrize(
    "_seconds, _minutes, _hours, result",
    [
      (1, 0, 0, "0:00:01"),
      (1, 0, 1, "1:00:01"),
      (1, 1, 0, "0:01:01"),
      (2, 3, 4, "4:03:02"),
    ]
)
def test_fix_elapsed(_seconds, _minutes, _hours, result):
  i: loguru.Record = {
    'elapsed':timedelta(seconds=_seconds, minutes=_minutes, hours=_hours),
    'exception':None,
    'extra': {},
    'file': None, #type: ignore
    'function': 'dummy_function',
    'level': None, #type: ignore
    'line': 23,
    'message': 'dummy_message',
    'module': 'smatter',
    'name': None,
    'process': None, #type: ignore
    'thread': None, #type: ignore
    'time': datetime.datetime(2020, 1, 1, 0, 0, 0, 0)
  }
  utils.fix_elapsed(i)
  assert i['extra']['elapsed'] == result

def test_close_queue_normal():
  q = mp.Queue()
  e = mp.Event()
  t = th.Thread(name='closer', target=utils.close_queue, args=(q, e, logger, False))
  try:
    q.put('dummy')
    t.start()
    time.sleep(0.5)
    assert t.is_alive()
    q.get()
    assert q.get() == None
    time.sleep(0.5)
    assert not t.is_alive()
  finally:
    e.set()
    t.join()
    q.join_thread()

def test_close_queue_fast():
  q = mp.Queue()
  e = mp.Event()
  t = th.Thread(name='closer', target=utils.close_queue, args=(q, e, logger, True))
  try:
    q.put('dummy')
    t.start()
    time.sleep(0.5)
    assert not t.is_alive()
  finally:
    e.set()
    t.join()
    q.join_thread()

def test_close_queue_stop():
  q = mp.Queue()
  e = mp.Event()
  t = th.Thread(name='closer', target=utils.close_queue, args=(q, e, logger, False))
  try:
    q.put('dummy')
    t.start()
    time.sleep(0.5)
    assert t.is_alive()
    e.set()
    time.sleep(0.5)
    assert not t.is_alive()
  finally:
    e.set()
    t.join()
    q.join_thread()

@pytest.mark.parametrize(
    "value",
    [
      b'dummy',
      b'\0\0\0\x18\x66\x74\x79\x70\x6d\x70',
      b'\x34\x32\0\0\0\0\x6d\x70\x34\x31',
      b'\x69\x73\x6f\x6d\0\0\0\x28\x75\x75',
    ]
)
def test_pipe_to_pipe(value):
  pipe_in = BytesIO(value)
  pipe_out = BytesIO()
  q = mp.Queue()
  e = mp.Event()
  t = utils.pipe_to_pipe(e, logger, 'dummy', 1, pipe_in, pipe_out)
  t.start()
  t.join()
  assert pipe_out.getvalue() == value

def test_pipe_to_mp_queue():
  pipe_in = BytesIO(b'dummy')
  q = mp.Queue()
  e = mp.Event()
  t = utils.pipe_to_mp_queue(e, logger, 'dummy', 1, pipe_in, q)
  t.start()
  assert q.get(True, 1) == b'd'
  assert q.get(True, 1) == b'u'
  assert q.get(True, 1) == b'm'
  assert q.get(True, 1) == b'm'
  assert q.get(True, 1) == b'y'
  assert q.get(True, 1) == None
  t.join(1)

def test_pipe_split():
  pipe_in = BytesIO(b'dummy')
  pipe_out = BytesIO()
  q = mp.Queue()
  e = mp.Event()
  t = utils.pipe_split(e, logger, 'dummy', 1, pipe_in, pipe_out, q)
  t.start()
  assert q.get(True, 1) == b'd'
  assert q.get(True, 1) == b'u'
  assert q.get(True, 1) == b'm'
  assert q.get(True, 1) == b'm'
  assert q.get(True, 1) == b'y'
  assert q.get(True, 1) == None
  assert pipe_out.getvalue() == b'dummy'
  t.join(1)