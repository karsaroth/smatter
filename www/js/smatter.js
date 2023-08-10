const vid = videojs('smatter-stream');
const loadedSubs = [];
var sub_buffer_pause = false;

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
      }
    } catch (e) {
      console.log(e);
    }
  },
  1000)

  //Set up video time monitor
  vid.on('progress', function() {
    try {
      const time = vid.currentTime();
      if (time < loadedSubs[loadedSubs.length - 1]) {
        vid.pause();
        sub_buffer_pause = true;
        var modal = player.createModal('Waiting for subtitles...');
      }
    } catch (e) {
      console.log(e);
    }
  });
});
