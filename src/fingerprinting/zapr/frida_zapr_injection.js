(function () {
    Date.prototype.timeNow = function () {
        return ((this.getHours() < 10)?"0":"") + this.getHours() +":"+ ((this.getMinutes() < 10)?"0":"") + this.getMinutes() +":"+ ((this.getSeconds() < 10)?"0":"") + this.getSeconds();
    }

    function print_args() {
        let str = "";
        for (let i = 0; i < arguments.length; i++) {
            str += arguments[i] + ", "
        }
        return str;
    }

    function log_with_date(logmessage) {
        let logstr = new Date().timeNow();
        logstr += ": ";
        logstr += logmessage;
        console.log(logstr);
    }

    rpc.exports = {
        init() {
            /*
            It seems like, when calling this init method, the Java.perform method does only work in the main process,
            but not in the subprocess. Why? Instead, we need to use Java.performNow, but need to be careful.
             */
            console.log("Called init");

            Java.performNow(function () {
                /*
                We use performNow, since perform does not seem to work in the subprocess :z_process.
                 */
                let AudioRecord = Java.use("android.media.AudioRecord");

                AudioRecord.$init.overload("int", "int", "int", "int", "int").implementation = function (audioSource, sampleRateInHz, channelConfig, audioFormat, bufferSizeInBytes) {
                    // Always log, when the audio recorder is accessed.
                    send("android.media.AudioRecord(" + this + ").<init>: " + print_args(audioSource, sampleRateInHz, channelConfig, audioFormat, bufferSizeInBytes));
                    return this.$init(audioSource, sampleRateInHz, channelConfig, audioFormat, bufferSizeInBytes);
                };
            });
            Java.performNow(function() {
                /*
                This does not seem to work in the subprocess (com.hindi.jagran.android.activity:z_process).
                Can we not access the whole application here? Only with performNow!
                 */

                let log_class = Java.use('com.redbricklane.zapr.basesdk.Log');
                console.log(new Date().timeNow() + " Found log class. Current value of logLevel enum:" + log_class.logLevel.value);
                console.log(new Date().timeNow() + " Current value of shouldWriteToLogFile:" + log_class.shouldWriteToLogFile.value);
                log_class.shouldWriteToLogFile.value = true;
                console.log(new Date().timeNow() + " New value of shouldWriteToLogFile:" + log_class.shouldWriteToLogFile.value);

                let LOG_LEVEL =  Java.use("com.redbricklane.zapr.basesdk.Log$LOG_LEVEL");
                log_class.logLevel.value = LOG_LEVEL.verbose.value;
                console.log(new Date().timeNow() + " New value of logLevel enum:" + log_class.logLevel.value);

                console.log("Overwriting individual log functions..");
                log_class.v.overload("java.lang.String", "java.lang.String").implementation = function(tag, message) {
                    log_with_date("V: " + tag + ":" + message);
                };
                log_class.setLogLevel.overload('com.redbricklane.zapr.basesdk.Log$LOG_LEVEL').implementation = function (log_level) {
                  // console.log(new Date().timeNow() + " Set log level was called.");
                  // console.log(new Date().timeNow() + " Calling original method with param: " + verbose_ll);
                  return log_class.setLogLevel(verbose_ll);
                };

                log_class.d.overload("java.lang.String", "java.lang.String").implementation = function(tag, message) {
                    log_with_date("D: " + tag + ":" + message);
                    log_class.d(tag, message);
                };
                log_class.i.overload("java.lang.String", "java.lang.String").implementation = function(tag, message) {
                    log_with_date("I: " + tag + ":" + message);
                    log_class.i(tag, message);
                };
                log_class.w.overload("java.lang.String", "java.lang.String").implementation = function(tag, message) {
                    log_with_date("W: " + tag + ":" + message);
                    log_class.w(tag, message);
                };
                log_class.e.overload("java.lang.String", "java.lang.String").implementation = function(tag, message) {
                    log_with_date("E: " + tag + ":" + message);
                    log_class.e(tag, message);
                };
            });
        },
        disablefingerprinting() {
            /*
            Disable automatic fingerprinting in background by hooking the start method.
            Automatic fingerprinting can disturb our own batch fingerprinting (can cause the script to exit too early).
             */
            Java.performNow(function() {
                console.log("Disabling automatic fingerprinting by hooking the fingerprint start method..");
                let FingerPrintHandler = Java.use("com.redbricklane.zapr.acrsdk.handlers.FingerPrintHandler");
                FingerPrintHandler.start.overload().implementation = function() {
                    return;
                };
            })
        },
        computefingerprint(audio_samples, algorithm_id, log) {
            if (log) {
                console.log("Called computefingerprint with " + audio_samples.length + " samples");
            }
            Java.performNow(function() {
                /*
                We expect the audio to have a length of a multiple of 8192 samples. The SDK is in practice only
                called with audio samples of this size (so multiples of 1024ms).
                 */
                if (audio_samples.length % 8192 !== 0) {
                    console.log("Error: Audio length is not a multiple of 8192 samples! It is: " + audio_samples.length);
                    send("error");
                    return;
                }
                let second_init_param = audio_samples.length;
                const JNI_CONST = 1576245358;
                let JNIConnectorCommonWrapper = Java.use('com.redbricklane.zapr.datasdk.jni.JNIConnectorCommonWrapper');
                let myJNIConnectorCommonWrapper = JNIConnectorCommonWrapper.$new();
                // Initialize the JNI connector and thereby the JNI (as in SDK)

                //let algorithm_id = 3; // Algorithm 3 is the "standard" algorithm, as submitted via JSON and set in code.
                if (log) {
                    console.log("Using algorithm id: " + algorithm_id);
                }
                myJNIConnectorCommonWrapper.init(algorithm_id, second_init_param, JNI_CONST);

                let int_fprint = myJNIConnectorCommonWrapper.process(audio_samples, audio_samples.length, JNI_CONST);
                if (int_fprint === null) {
                    console.log("Generated fingerprint is null!");
                    send("error");
                    return;
                }
                if (log) {
                    console.log("second_init_param: " + second_init_param);
                    console.log("Int fprint has length " + int_fprint.length);
                }

                let AcrSDKUtility = Java.use('com.redbricklane.zapr.acrsdk.util.AcrSDKUtility');

                let byte_fprint = AcrSDKUtility.getFingerPrintByteArray(int_fprint);
                if (log) {
                    console.log("Byte fprint has length " + byte_fprint.length);
                }
                let fprint_bytearr = new Int8Array(byte_fprint.length);
                for (var i = 0; i < byte_fprint.length; ++i) {
                    /*
                    Write to a buffer-like object so it can be sent with Frida
                     */
                    fprint_bytearr[i] = byte_fprint[i];
                }
                // Now deinit the JNI connector
                if (log) {
                    console.log("Now deinitializing the JNI.");
                }
                myJNIConnectorCommonWrapper.deInit(JNI_CONST);
                send("fingerprint", fprint_bytearr);
                /*
                Returning a value from here does not seem to work (at least not for a byte array). We need to
                use the messaging via 'send'.
                 */
            });
        }
    };
}).call(this);