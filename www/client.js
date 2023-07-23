// peer connection
var pc = null;

function createPeerConnection() {
  var config = {
    sdpSemantics: 'unified-plan'
  };
  pc = new RTCPeerConnection(config);
  // connect audio / video
  pc.addEventListener('track', function(evt) {
    console.log(`Got track event ${JSON.stringify(evt)}`)
    if (evt.track.kind == 'video')
      document.getElementById('video').srcObject = evt.streams[0];
    else
      document.getElementById('audio').srcObject = evt.streams[0];
  });
  pc.addEventListener('datachannel', function(evt) {
    console.log(`Got datachannel event ${evt}`)
    textTrack = document.getElementById('video').appendChild(document.createElement('track', {
      kind: 'subtitles',
      default: true
    }));
    evt.channel.addEventListener('message', function(evt) {
      smatterSub = JSON.parse(evt.data)
      cue = VTTCue(smatterSub.start, smatterSub.end, smatterSub.text)
      textTrack.addCue(cue)
    });
  });

  return pc;
}

function negotiate() {
  pc.addTransceiver('video', {direction: 'recvonly'});
  pc.addTransceiver('audio', {direction: 'recvonly'});
  return pc.createOffer().then(function(offer) {
    return pc.setLocalDescription(offer);
  }).then(function() {
    var offer = pc.localDescription;

    return fetch('/offer', {
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type
      }),
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST'
    });
  }).then(function(response) {
      return response.json();
  }).then(function(answer) {
      return pc.setRemoteDescription(answer);
  }).catch(function(e) {
      alert(e);
  });
}

function start() {
  pc = createPeerConnection();
  negotiate();
}

start();