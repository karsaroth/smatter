# smatter

**Smatter** provides real time, local translation of livestreams (on the viewer side).
Currently it is in a very basic state, and has only been tested on windows, but seems initially usable for youtube and twitch streams. It relies primarily on [faster-whisper](https://github.com/guillaumekln/faster-whisper) for translation, with some help from [silero_vad](https://github.com/snakers4/silero-vad).

## Dependencies

You will need binaries for these applications installed for Smatter to function:

* [yt-dlp](https://github.com/yt-dlp/yt-dlp)
* [ffmpeg](https://ffmpeg.org)

And others are recommended too:

* [mpv](https://mpv.io/) - required to use 'watch' output.
* [phantomJS](https://phantomjs.org/) - yt-dlp recommends this to avoid throttling issues
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) - Will allow faster-whisper to use an Nvidia GPU. See [CTranslate2](https://opennmt.net/CTranslate2/installation.html) for details.

## Installation

1. Check out or download the latest code
2. Add required libraries using `pip install -r .\requirements.txt`
3. Install dependencies, and add them to your PATH (or to ./libs/bin directory)
4. Add mpv-2.dll to ./libs/bin folder on windows. See [here](https://sourceforge.net/projects/mpv-player-windows/files/libmpv/).
5. Pre-downloading faster-whisper model(s) is also recommended, the application doesn't wait for this process currently, and will probably fail.

## Command Line Arguments
```
usage: main.py [-h] --source SOURCE [--quality QUALITY] [--start START] --output {srt,vtt,watch} [--output-dir OUTPUT_DIR] [--output-file OUTPUT_FILE]
               [--model-size {tiny,base,small,medium,large,large-v2,tiny.en,base.en,small.en,medium.en}] [--force-gpu] [--source-language SOURCE_LANGUAGE] 
               [--goal {translate,transcribe}] [--log-level {debug,info,warning,error,critical,none}]

options:
  -h, --help            show this help message and exit
  --source SOURCE       URL of stream/video
  --quality QUALITY     Max vertical video size (e.g 480 for 480p). If not specified, the best possible quality is chosen.
  --start START         Start point of a vod in HH:mm:ss (defaults to 0:00:00)
  --output {srt,vtt,watch}
                        What output format is desired (file or video window)
  --output-dir OUTPUT_DIR
                        Directory to store output files (defaults to ./output)
  --output-file OUTPUT_FILE
                        Filename for any output file (defaults to output.srt)
  --model-size {tiny,base,small,medium,large,large-v2,tiny.en,base.en,small.en,medium.en}
                        Whisper model selection (defaults to base)
  --force-gpu           Force using GPU for translation (requires CUDA
                        libraries). Shouldn't be necessary in most cases.
  --source-language SOURCE_LANGUAGE
                        Source language short code (defaults to en for English)
  --goal {translate,transcribe}
                        Select between translation or transcription (defaults to transcription)
  --log-level {debug,info,warning,error,critical,none}
                        How much log info to show in command window (defaults to warning)
```

## Examples
### Minimum configuration
**Open video window with transcribed english speech shown along with video stream**
(NOTE: currently, quality default of 'best' seems to cause problems for MPV, possibly issues with 60fps?)
```
python main.py --source https://www.youtube.com/watch?v=lKDZ_hmDqMI --output watch --quality 480
```

**Save srt file to be used with separate application later**
```
python main.py --source https://www.youtube.com/watch?v=lKDZ_hmDqMI --output srt
```

**Save webvtt file instead**
```
python main.py --source https://www.youtube.com/watch?v=lKDZ_hmDqMI --output vtt
```

(NOTE: srt and webvtt files can be used with browser plugins for vods)

### Translate to English
**Watch video stream**
(NOTE: A Larger model seems to do better at translation.)
```
python main.py --source https://www.youtube.com/watch?v=jjiXgRO8qDw --output watch --quality 480 --goal translate --source-language it --model-size large-v2
```

### Start later into a video stream
**Watch video stream**
```
python main.py --source https://www.youtube.com/watch?v=D_DtKgsr9WQ --output watch --quality 480 --goal translate --source-language ja --model-size large-v2 --start 0:08
```

## Notes on Translation/Transcription
* Three confidence markers are prefixed to each line (e.g. `[---]`). They represent, in order:
  * Probability (from Whisper's log_probablity): - `-`, `?` and `!` represent high to low respectively.
  * No Speech (Noise) probability: `-`, `?` and `!` represent low to high respectively.
  * Compression: `-`, `?` and `!` represent low to high respectively.
  If you're uncertain what this means, a summary is that [---] is likely to be fairly accurate, [!!!] is likely to be very innacurate.
* Translation works reasonably well with background music or sound, so long as it isn't too loud compared to the speech.
* Singing or speaking with unusual speech patterns may produce poor results.
* Multiple speakers will often be translated well, but the output will not differentiate between them.
* The Whisper models will sometimes produce repeated false translations. You can filter these with `gigo_phrases.txt` for now.


## Future plans
If possible, I would like to add these features in the future:
* Restreaming, primarily to allow the use of video players other than MPV, such as a mobile device.
* Separating the translation component, so it can be run remotely, or using a docker container.
* Better handling of various scenarios (seeking, disconnection, upcoming streams, etc).
* A nicer interface, and easier installation (of app and dependencies).
* Streamer side translation (OBS Plugin?)
* Other nice things (e.g. more destination languages, other models, easier installation, etc)