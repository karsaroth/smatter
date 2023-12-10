"""Exposes Faster Whisper over WebRTC using aiortc."""

from __future__ import annotations
import argparse
from pathlib import Path
import queue
import string
import os
import sys
import threading
from typing import List, Optional

import loguru
from loguru import logger

from smatter.transx import InteractiveTransXProcess, TransXConfig, WhisperConfig, transx_from_socket_server
from smatter.smatter_socket import run_proccess_server

ROOT = os.path.dirname(__file__)

def fix_elapsed(record: loguru.Record):
  """
  Fixes the elapsed value in log messages to display appropriately
  """
  record["extra"]["elapsed"] = str(record["elapsed"])

logger.remove()
logger.configure(patcher=fix_elapsed)
logger.add(
  sys.stderr, # type: ignore
  format="<g>{extra[elapsed]}</g> | <level>{level: <8}</level> | <c>{process.name}</c>:<c>{thread.name}</c>:<c>{process.id}</c>:<c>{function}</c>:<c>{line}</c> - <level>{message}</level>",
  level="INFO",
  colorize=True,
  enqueue=True,
  backtrace=True,
  diagnose=True
)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Smatter TCP loopback"
  )
  parser.add_argument(
      "-a", "--host", default="localhost", help="Host for TCP listener (default: localhost)"
  )
  parser.add_argument(
      "-p", "--port", type=int, default=9999, help="Port for TCP listener (default: 9999)"
  )
  parser.add_argument(
    "-m", "--model-size", 
    type=str,
    default="medium.en",
    help="""Whisper model selection.
            By default this is medium.en,
            best for transcribing english.
            Larger models may be more accurate,
            but may also be too slow for some 
            systems, or require too much 
            RAM or GPU VRAM.""",
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
  parser.add_argument("-l", "--source-language",
    type=str,
    default="en",
    help="""Source language short code (e.g. 'en' = English,
         'ja' = Japanese, 'de' = German). Default is 'en'.
         See ISO_639-1 for a complete list, however the
         Whisper model may not support all languages."""
  )
  parser.add_argument("goal",
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
  parser.add_argument("-s", "--use-process",
    help="""Use a separate process for translation.
            Default is false, but can be enabled if
            single-core processing isn't fast enough.""",
    action='store_true'
  )
  args = parser.parse_args()

  transx: Optional[InteractiveTransXProcess] = None

  try:
    # Read default gigo if available
    gigo_file = Path('./gigo_phrases.txt')
    gigo_phrases: List[str] = []
    if gigo_file.exists():
      logger.info('Loading gigo phrases from gigo.txt')
      with gigo_file.open('r', encoding='UTF-8', ) as gigo:
        gigo_phrases = gigo.readlines()
    gigo_phrases = [
      phrase.strip()
            .translate(str.maketrans('', '', string.punctuation))
            .casefold()
      for phrase in gigo_phrases
    ]
    logger.info('gigo phrases: {gigo}', gigo=gigo_phrases)

    default_lang = 'en'
    default_goal = 'transcribe'
    if hasattr(args, 'source_language'):
      default_lang = args.source_language
    if hasattr(args, 'goal'):
      default_goal = args.goal

    whisper_config = WhisperConfig(
      model = 'faster_whisper',
      model_size = args.model_size,
      force_gpu = args.force_gpu,
      lang = default_lang,
      goal = default_goal,
      model_root = args.cache_dir,
      gigo_phrases=gigo_phrases
    )

    if args.use_process:
      logger.info('Using separate process for translation.')
      transx = InteractiveTransXProcess(
        logger,
        whisper_config
      )
      transx.check_model()
      transx.start(
        '0',
        default_lang,
        default_goal
      )

      run_proccess_server(
        transx.input_queue,
        transx.output_queue,
        args.host,
        args.port,
        logger
      )
    else:
      logger.info('Using threads for translation.')

      transx_config = TransXConfig(
        _logger = logger,
        stop = threading.Event(),
        output_queue= queue.Queue(),
        model_config=whisper_config,
        format='plain',
        requested_start='0',
        base_path='',
        stream_url='',
      )

      transx_from_socket_server(
        transx_config,
        default_lang,
        default_goal,
        args.host,
        args.port
      )

  except KeyboardInterrupt:
    logger.info("Keyboard interrupt, exiting.")
    if transx is not None:
      transx.stop()
  except Exception as e:
    logger.exception("Unexpected error caught in main.")
    if transx is not None:
      transx.stop()

  logger.info("Smatter exiting.")
