/*
Frida script that waits for a message containing an audio buffer (bytes) and then computes Deezer's
fingerprint (with or without encryption, depending on the variable USE_ENCRYPTION).
The fingerprint is returned to the calling script.
It works with Deezer Android app 6.1.14.99, but with one small change (see code) also for 6.2.13.151 (more recent).
The access secret is hardcoded into the app, the ekey was the same over multiple months on multiple devices.
 */
setTimeout(function(){
  Java.perform(function() {
    //var ACRCloudRecognizer = Java.use("com.acrcloud.rec.engine.ACRCloudRecognizeEngine"); // version 6-
    var ACRCloudRecognizer = Java.use("com.acrcloud.rec.engine.ACRCloudUniversalEngine"); // version 7-

    var EKEY = "D2D7D34BB80628B57EB51C7F9CC44698";
    var ACCESS_SECRET = "uBnEA6VQ1gn1SF6JTWiOxIN45C1PM0vXcrVotPSR";
    const USE_ENCRYPTION = true;
    /*
    If USE_ENCRYPTION is not set, the fingerprint generation method is called with the null parameters.
    Otherwise, it is called with EKEY and ACCESS_SECRET for encryption
     */
    if (!USE_ENCRYPTION) {
      EKEY = null;
      ACCESS_SECRET = null;
      console.log("Encryption of fingerprints disabled!");
    }
    else {
      console.log("Encryption of fingerprints enabled!");
    }

    while (true) {
      var wav_bytes;
      recv('wave', function (message) {
        // Receive the next wav bytes from the python process. Wait for that.
        // message.payload contains the audio
        if (message.payload === null) {
          // This indicates that all files have been processed
          wav_bytes = null;
          return;
        }
        //console.log("Received audio samples in javascript of length " + message.payload.length + " bytes");

        // now create something that can be fed into the fingerprinting method
        wav_bytes = new Int8Array(message.payload.length);
        for (var i = 0; i < message.payload.length; ++i) {
          wav_bytes[i] = message.payload[i];
        }
      }).wait();
      if (wav_bytes === null) {
        console.log("Received null wav. Exiting.");
        break;
      }
      /*
      Note: To target Deezer v 6.2.13.151 instead of 6.1.14.99, simply replace the method call to 'b' (instead of 'a')
       */
      // var fingerprint = ACRCloudRecognizer.a(wav_bytes, wav_bytes.length, EKEY, ACCESS_SECRET, 100, false); // version 6-
      var fingerprint = ACRCloudRecognizer.native_create_fingerprint(wav_bytes, wav_bytes.length, 100, EKEY, ACCESS_SECRET, false); // version 7-

      if (fingerprint === null) {
        //console.log("Generation error. Sending error from JS");
        send('error');
        continue;
      }
      var fprint_bytearr = new Int8Array(fingerprint.length);
      for (var i = 0; i < fingerprint.length; ++i) {
        /*
        Write to a buffer-like object so it can be sent with Frida
         */
        fprint_bytearr[i] = fingerprint[i];
      }
      send('fingerprint', fprint_bytearr);
    }
  });
}, 0);
