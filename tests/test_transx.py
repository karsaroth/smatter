from loguru import logger
from smatter import transx
from faster_whisper.transcribe import Segment
from libs.vad.utils_vad import VADIterator
from pytest_mock import MockerFixture
import io
import numpy as np
import pytest
import string
import multiprocessing as mp


def generate_txdata(text) -> transx.TransXData:
  return {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.1,
    "noise_probability": 0.2,
    "compression_ratio": 0.3,
    "text": text.strip()
  }

@pytest.mark.parametrize(
    "base_start, start, end, text, prob, noise, comp",
    [
      (5.0, 0.423, 0.555, "Hello everyone!", -0.522878745280338, 0.1, 1.2),
      (10.4, 0.0, 1.0, "Were you listening to me?", -0.154901959985743, 0.1, 1.5),
      (123.12, 4.99, 5.94, "That's the plan for today.", -0.638272163982407, 0.1, 0.8),
      (1423.1, 12.9, 13, "That doesn't really make sense...", -3.88605664769316, 0.7, 3.2),
    ]
)
def test_segment_to_txdata(base_start, start, end, text, prob, noise, comp):
  s = Segment(
    1,
    16430,
    start,
    end,
    text,
    [15496, 11075, 0],
    0.0,
    prob,
    comp,
    noise,
    None,
  )
  result = transx.segment_to_txdata(s, base_start)
  assert result == {
    "start": base_start + start,
    "end": base_start + end,
    "probability": np.exp(prob),
    "noise_probability": noise,
    "compression_ratio": comp,
    "text": text.strip()
  }

@pytest.mark.parametrize(
  "input, output, fixed",
  [
    ("That's crazy, that's crazy, that's crazy, that's crazy, that's crazy, that's crazy!",
     "That's crazy, that's crazy, that's crazy[...]",
     True),
    ("Honestly I don't know what to say, I don't know what to say, I don't know what to say, I don't know what to say, I don't know what to say.",
     "Honestly I don't know what to say, I don't know what to say, I don't know what to say[...]",
     True),
    ("There might be something in here? Oh no! oh no! oh no! oh no! oh no! oh no! Run!",
     "There might be something in here? Oh no! oh no! oh no! oh no! oh no! oh no! Run!",
     False),
    ("There might be something in here? Oh no! oh no! oh no! oh no! oh no! oh no!",
     "There might be something in here? Oh no! oh no! oh no![...]",
     True),
    ("If there's a point to this, perhaps it could be... I don't know if there's a point to this? Do you think there's a point to this? I don't think there's a point to this.",
     "If there's a point to this, perhaps it could be... I don't know if there's a point to this? Do you think there's a point to this? I don't think there's a point to this.",
     False),     
  ],
)
def text_fix_repeated_phrases(input, output, fixed):
  result = transx.fix_repeated_phrases(input)
  assert result == (output, fixed)

@pytest.mark.parametrize(
  "input, output, fixed",
  [
    ("Hahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahahaha",
     "Hahahahahahaha[...]",
     True),
    ("Nonononononononono",
     "Nonononononono[...]",
     True),
    ("That looks amazing! Wowowowowowowowowowowowowowowowowow",
     "That looks amazing! Wowowowowowow[...]",
     True),    
    ("HAHAHAHAHAHAHA",
     "HAHAHAHAHAHA[...]",
     True),
    ("HAHAHAHAHAHA",
     "HAHAHAHAHAHA",
     True),
    ("HAHAHAHAHA",
     "HAHAHAHAHA",
     False),
    ("HA HA HA HA HA HA HA",
     "HA HA HA HA HA HA HA",
     False)
  ],
)
def test_fix_repeated_sounds(input, output, fixed):
  result = transx.fix_repeated_sounds(input)
  assert result == (output, fixed)

@pytest.mark.parametrize(
  "input, size, first",
  [
    ([
      "Bye! Bye!",
      "Hello!",
      "Thanks for watching my last video"
    ],
    2,
    "Hello!"),
    ([
      "Goodbye!",
      "please subscribe to my channel",
      "SEE YOU IN THE NEXT VIDEO?"
    ],
    1,
    "Goodbye!"),    
  ],
)
def test_filter_gigo_results(input, size, first):
  tx_data = list(map(lambda x: generate_txdata(x), input))
  test_gigo_phrases = list(map(lambda x: x.translate(str.maketrans('', '', string.punctuation)).casefold(), [
    'Bye Bye',
    'Please subscribe to my channel',
    'Thanks for watching',
    'See you in the next video'
  ]))
  result = transx.filter_gigo_results(tx_data, test_gigo_phrases)
  assert len(result) == size
  assert result[0]["text"] == first

@pytest.mark.parametrize(
  "input, output, vtt",
  [
    (1.0, "00:00:01,000", False),
    (61.0, "00:01:01,000", False),
    (2.123456789, "00:00:02,123", False),
    (86399.999, "23:59:59,999", False),
    (1.3, "00:00:01.300", True),
    (543.5678, "00:09:03.567", True) 
  ],
)
def test_seconds_to_timestamp(input, output, vtt):
  result = transx.seconds_to_timestamp(input, vtt)
  assert result == output

def test_transx_to_string(monkeypatch):
  monkeypatch.setattr(transx, "fix_repeated_phrases", lambda x: (x, True))
  monkeypatch.setattr(transx, "fix_repeated_sounds", lambda x: (x, True))
  tx: transx.TransXData = {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.9,
    "noise_probability": 0.1,
    "compression_ratio": 1.0,
    "text": 'Sample text'
  }
  result = transx.transx_to_string(tx)
  assert result == '[---]: Sample text'

  tx1: transx.TransXData = {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.4,
    "noise_probability": 0.6,
    "compression_ratio": 2.1,
    "text": 'Sample text'
  }
  result1 = transx.transx_to_string(tx1)
  assert result1 == '[???]: Sample text'

  tx2: transx.TransXData = {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.2,
    "noise_probability": 0.9,
    "compression_ratio": 3.2,
    "text": 'Sample text'
  }
  result2 = transx.transx_to_string(tx2)
  assert result2 == '[!!!]: Sample text'
  
def test_txdata_to_srt(monkeypatch):
  monkeypatch.setattr(transx, "seconds_to_timestamp", lambda x, y: "00:00:01,000" if not y else "00:00:01.000")
  monkeypatch.setattr(transx, "transx_to_string", lambda x: '[---]: Sample text')
  tx: transx.TransXData = {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.9,
    "noise_probability": 0.1,
    "compression_ratio": 1.0,
    "text": 'Sample text'
  }
  result = transx.txdata_to_srt(tx, 1, False)
  assert result == '1\n00:00:01,000 --> 00:00:01,000\n[---]: Sample text\n\n'
  result1 = transx.txdata_to_srt(tx, 1, True)
  assert result1 == '1\n00:00:01.000 --> 00:00:01.000\n[---]: Sample text\n\n'

def test_join_similar():
  tx1: transx.TransXData = {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.1,
    "noise_probability": 0.2,
    "compression_ratio": 0.3,
    "text": 'Sample text'
  }
  tx2: transx.TransXData = {
    "start": 1.0,
    "end": 2.0,
    "probability": 0.1,
    "noise_probability": 0.2,
    "compression_ratio": 0.3,
    "text": 'Sample text'
  }
  tx3: transx.TransXData = {
    "start": 3.0,
    "end": 4.0,
    "probability": 0.9,
    "noise_probability": 0.3,
    "compression_ratio": 1.0,
    "text": 'More sample text'
  }
  tx4: transx.TransXData = {
    "start": 5.0,
    "end": 6.0,
    "probability": 0.5,
    "noise_probability": 0.1,
    "compression_ratio": 2.0,
    "text": 'More sample text'
  }
  result = transx.join_similar([tx1, tx2, tx3, tx4])
  assert len(result) == 2
  assert result[0]["text"] == 'Sample text'
  assert result[0]["start"] == 1.0
  assert result[0]["end"] == 2.0
  assert result[0]["probability"] == 0.1
  assert result[0]["noise_probability"] == 0.2
  assert result[0]["compression_ratio"] == 0.3
  assert result[1]["text"] == 'More sample text'
  assert result[1]["start"] == 3.0
  assert result[1]["end"] == 6.0
  assert result[1]["probability"] == 0.9
  assert result[1]["noise_probability"] == 0.3
  assert result[1]["compression_ratio"] == 2.0

class MockProcess:
  _inbytes = b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'
  stdout = io.BytesIO(_inbytes)

#Testing transx.chunk_from_samples()
def text_chunk_from_samples():
  mock_process = MockProcess()
  chunk_gen = transx.chunk_from_samples(mp.Event(), logger, mock_process, 2) #type: ignore
  result = np.zeros(0, np.float32)
  for c in chunk_gen:
    assert isinstance(c, np.ndarray)
    assert c.dtype == np.float32
    assert c.shape == (2,)
    result += c
  assert result == np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0])

def test_vad_samples(mocker: MockerFixture):
  count = 0
  def pretend_vad(self, x, return_seconds=False):
    nonlocal count
    assert x.shape == (2,)
    assert x.dtype == np.float32
    assert x.all() == np.array([-12.3, 4.9], np.float32).all()
    count += 1
    if count == 2:
      return {'start': 3}
    elif count == 3:
      return {'end': 6}
    else:
      return None

  mocker.patch('libs.vad.utils_vad.VADIterator.reset_states')
  mocker.patch('libs.vad.utils_vad.VADIterator.__call__', pretend_vad)
  chunk_size = 2
  max_size = 10
  start = 0
  blanks = mp.Queue()
  def chunk_gen():
    for x in range(1, 10):
      yield np.array([-12.3, 4.9], np.float32)

  vad = transx.vad_samples(logger, chunk_gen(), chunk_size, max_size, start, blanks)
  first_start, first = vad.__next__()
  assert first_start == 3
  assert first.all() == np.array([-12.3, 4.9, -12.3, 4.9, -12.3, 4.9], np.float32).all()

  with pytest.raises(StopIteration):
    vad.__next__()

def test_vad_samples_maxed(mocker: MockerFixture):
  count = 0
  def pretend_vad(self, x, return_seconds=False):
    nonlocal count
    assert x.shape == (2,)
    assert x.dtype == np.float32
    assert x.all() == np.array([-12.3, 4.9], np.float32).all()
    count += 1
    if count == 1:
      return {'start': 1}
    else:
      return None

  mocker.patch('libs.vad.utils_vad.VADIterator.reset_states')
  mocker.patch('libs.vad.utils_vad.VADIterator.__call__', pretend_vad)
  chunk_size = 2
  max_size = 10
  start = 0
  blanks = mp.Queue()
  def chunk_gen():
    for x in range(1, 10):
      yield np.array([-12.3, 4.9], np.float32)

  vad = transx.vad_samples(logger, chunk_gen(), chunk_size, max_size, start, blanks)
  first_start, first = vad.__next__()
  assert first_start == 1
  assert first.all() == np.array([-12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9, -12.3, 4.9], np.float32).all()

def test_vad_samples_silences(mocker: MockerFixture):
  sample = 0
  def pretend_vad(self, x, return_seconds=False):
    nonlocal sample
    assert x.shape == (1024,)
    assert x.dtype == np.float32
    sample += len(x)
    self.current_sample = sample
    return None

  mocker.patch('libs.vad.utils_vad.VADIterator.reset_states')
  mocker.patch('libs.vad.utils_vad.VADIterator.__call__', pretend_vad)

  chunk_size = 1024
  start = 0
  blanks = mp.Queue()
  def chunk_gen():
    for x in range(1, 72):
      yield np.zeros(1024, np.float32)
  
  vad = transx.vad_samples(logger, chunk_gen(), chunk_size, 320000, start, blanks)

  with pytest.raises(StopIteration):
    vad.__next__()
  
  assert blanks.qsize() == 4
  assert blanks.get() == (0.0, 1.024, None)
  assert blanks.get() == (1.088, 2.112, None)
  assert blanks.get() == (2.176, 3.2, None)
  assert blanks.get() == (3.264, 4.288, None)