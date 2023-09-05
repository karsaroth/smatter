const vid = videojs('smatter-stream');
const sourceButton = document.getElementById('source-update');
const startButton = document.getElementById('smatter-toggle');
const loadedSubs = [];
var sub_buffer_pause = false;
var modal;

function switchBooleanIndicator(bool, element) {
  if (bool) {
    element.classList.remove('bi-x-circle-fill');
    element.classList.add('bi-check-circle-fill');
    element.classList.remove('text-warning');
    element.classList.add('text-success');
  } else {
    element.classList.remove('bi-check-circle-fill');
    element.classList.add('bi-x-circle-fill');
    element.classList.remove('text-success');
    element.classList.add('text-warning');
  }
}

sourceButton.addEventListener('click', async function() {
  const source = document.getElementById('source-url').value;
  const quality = document.getElementById('quality').value;
  const startPoint = document.getElementById('start-point').value;
  let sourceLanguage = document.getElementById('language').value;
  const task = document.getElementById('task').value;
  //Check if source language matches a suggestion format (e.g. 'English (en)') 
  //using regex and if so, extract the language code from the suggestion
  const sourceLanguageRegex = /.*\((.*)\)/;
  const sourceLanguageMatch = sourceLanguage.match(sourceLanguageRegex);
  if (sourceLanguageMatch !== null) {
    sourceLanguage = sourceLanguageMatch[1];
  }

  req = {
    "action": "set",
    "stream_url": source,
    "language": sourceLanguage,
    "requested_start": startPoint,
    "goal": task,
    "quality": quality
  }

  for (const [key, value] of Object.entries(req)) {
    if (value === "") {
      delete req[key];
    }
  };

  const response = await fetch('/smatter/state', {
    method: 'POST',
    body: JSON.stringify(req),
    headers: {
      'Content-Type': 'application/json'
    }
  });

  if (!response.ok) {
    console.log('State update failed');
  }
});

startButton.addEventListener('click', async function() {
  action = "stop"
  if (startButton.classList.contains('active')) {
    action = "start"
  }
  const response = await fetch('/smatter/state', {
    method: 'POST',
    body: JSON.stringify({"action": action}),
  });
  if (!response.ok) {
    console.log('Start/Stop request failed');
    if (action === "start") {
      startButton.classList.remove('active');
    } else {
      startButton.classList.add('active');
    }
  }
});


vid.ready(function() {
  //Set up subtitle monitor
  setInterval(async function() {
    try {
      const textTracks = vid.textTracks();
      const textTrack = textTracks.tracks_.find(track => track.id === 'smatter');
      const response = await fetch('/smatter/subtitles', {
        method: 'GET'
      });
      const smatterSubList = await response.json();
      smatterSubList.forEach(smatterSub => {
        if (!loadedSubs.includes(smatterSub.start)) {
          loadedSubs.push(smatterSub.start);
          if (smatterSub.text !== undefined && smatterSub.text !== null) {
            cue = new VTTCue(smatterSub.start, smatterSub.end, smatterSub.text);
            textTrack.addCue(cue);
          }
        }
      });
      if (sub_buffer_pause) {
        vid.play();
        sub_buffer_pause = false;
        if (modal !== undefined) {
          modal.close();
          modal = undefined;
        }
      }
    } catch (e) {
      console.log(e);
    }
  },
  1000);

  //Set up video time monitor
  vid.on('progress', function() {
    try {
      const time = vid.currentTime();
      if (time >= loadedSubs[loadedSubs.length - 1]) {
        console.log(`Current Time: ${time} >= Loaded Sub's Latest: ${loadedSubs[loadedSubs.length - 1]}, pausing.`)
        vid.pause();
        sub_buffer_pause = true;
        if (modal == undefined) {
          modal = vid.createModal('Waiting for subtitles...');
        }
      }
    } catch (e) {
      console.log(e);
    }
  });
});

//Update status indicators and start button
setInterval(async function() {
  modelIndicator = document.getElementById('ind-model');
  streamIndicator = document.getElementById('ind-stream');
  subsIndicator = document.getElementById('ind-subs');
  transxIndicator = document.getElementById('ind-transx');
  startReady = true;

  try {
    const response = await fetch('/smatter/status', {
      method: 'GET'
    });
    const smatterStatusData = await response.json();
    // Status Data structure:
    // {
    //   "model": true,
    //   "stream": true,
    //   "transx_state": "running",
    //   "logs": ["A log message", "Another log message"]
    // }
    switchBooleanIndicator(smatterStatusData.model, modelIndicator);
    switchBooleanIndicator(smatterStatusData.stream, streamIndicator);
    if (smatterStatusData.model && smatterStatusData.stream) {
      startButton.classList.remove('disabled');
    } else {
      startButton.classList.add('disabled');
    }
    
    if (smatterStatusData.transx_state === 'running') {
      transxIndicator.classList.remove('text-success');
      transxIndicator.classList.add('text-info');
      transxIndicator.classList.remove('bi-stop-circle-fill');
      transxIndicator.classList.add('bi-play-circle-fill');
    }
    else if (smatterStatusData.transx_state === 'stopped') {
      transxIndicator.classList.remove('text-success');
      transxIndicator.classList.add('text-info');
      transxIndicator.classList.remove('bi-play-circle-fill');
      transxIndicator.classList.add('bi-stop-circle-fill');
    }
    else {
      transxIndicator.classList.remove('text-info');
      transxIndicator.classList.add('text-success');
      transxIndicator.classList.remove('bi-play-circle-fill');
      transxIndicator.classList.add('bi-stop-circle-fill');
    }
    
    if (loadedSubs.length > 0) {
      switchBooleanIndicator(true, subsIndicator);
    } else {
      switchBooleanIndicator(false, subsIndicator);
    }

  } catch (e) {
    console.log(e);
    switchBooleanIndicator(false, modelIndicator);
    switchBooleanIndicator(false, modelIndicator);
  }
},
1500);

//Trigger model loading
//(Will be more useful once dynamic models are possible)
window.onload = async function() {
  try {
    const response = await fetch('/smatter/state', {
      method: 'POST',
      body: JSON.stringify({"action": "load"}),
      headers: {
        'Content-Type': 'application/json'
      }
    });
    const responseStatus = response.status;
    console.log(`Load Status was ${responseStatus}`);
  } catch (e) {
    console.log(e);
  }
};