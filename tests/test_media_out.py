from __future__ import annotations
from pathlib import Path
from unittest.mock import call
from smatter import media_out
from loguru import logger
from pytest_mock import MockerFixture
import multiprocessing as mp
import pytest
import pytest_mock
import builtins

@pytest.fixture
def mocker_open(mocker: MockerFixture):
  mock_ret = mocker.mock_open()
  mocker.patch('builtins.open', mock_ret)
  return mock_ret

def test_save_srt(mocker_open):
  e = mp.Event()
  q = mp.Queue()
  q.put((0, 3, 'one\n'))
  q.put((5, 7, 'two\n'))
  q.put((9, 13, 'three\n'))
  q.put((19, 21, 'four\n'))
  q.put((25, 26, 'five\n'))
  q.put(None)

  media_out.save_srt(
      e,
      logger,
      q,
      False,
      Path('./dummy'),
      'srt_dummy.srt'
  )

  mocker_open.assert_has_calls([
    call((Path('./dummy') / 'srt_dummy.srt').absolute().as_posix(), 'w', encoding="utf-8", newline='\n'),
    call().__enter__(),
    call().write('one\n'),
    call().flush(),
    call().write('two\n'),
    call().flush(),
    call().write('three\n'),
    call().flush(),
    call().write('four\n'),
    call().flush(),
    call().write('five\n'),
    call().flush(),
    call().__exit__(None, None, None)
  ])

def test_save_srt_with_vtt(mocker_open):
  e = mp.Event()
  q = mp.Queue()
  q.put((0, 3, 'one\n'))
  q.put((5, 7, 'two\n'))
  q.put((9, 13, 'three\n'))
  q.put((19, 21, 'four\n'))
  q.put((25, 26, 'five\n'))
  q.put(None)

  media_out.save_srt(
      e,
      logger,
      q,
      True,
      Path('./dummy'),
      'srt_dummy.webvtt'
  )

  mocker_open.assert_has_calls([
    call((Path('./dummy') / 'srt_dummy.webvtt').absolute().as_posix(), 'w', encoding="utf-8", newline='\n'),
    call().__enter__(),
    call().write('WEBVTT\n\n'),
    call().flush(),
    call().write('one\n'),
    call().flush(),
    call().write('two\n'),
    call().flush(),
    call().write('three\n'),
    call().flush(),
    call().write('four\n'),
    call().flush(),
    call().write('five\n'),
    call().flush(),
    call().__exit__(None, None, None)
  ])