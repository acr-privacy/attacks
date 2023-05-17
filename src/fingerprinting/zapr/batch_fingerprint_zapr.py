"""
Python script to batch fingerprint wav files using the Zapr SDK 6.1.1 .
The script uses the corresponding Frida injection script 'frida_zapr_injection.js'
After starting the script, the com.winit.starnews.hin app needs manual force closing and then a manual start.
Make sure that the device is connected to the internet, otherwise it might not start the z_process, which we hook.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from glob import glob

import frida
import logging
import numpy as np
import time
import threading

sys.path.append(os.getcwd())

from audio.utils import resample
from audio.io.audioReader import readWAV
from common.utils import compute_outfile, fingerprint_to_file, fingerprint_to_string


CURRENT_FILE = ''
FINGERPRINT_IN_PROGRESS = False  # indicates whether the python script should wait for the next fingerprint
ARGS = None
GOT_ERROR = False

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def on_message(message, data):
    """
    Message handler that receives the generated fingerprint from the Frida script.
    It is then stored to a file.
    """
    global FINGERPRINT_IN_PROGRESS
    global CURRENT_FILE
    global ARGS

    if 'payload' not in message or message['payload'] == 'error':
        logging.error(f"No fingerprint generated for {CURRENT_FILE}")
        FINGERPRINT_IN_PROGRESS = False
        return
    if message['payload'] == 'fingerprint':
        fingerprint = data  # data contains the fingerprint as binary data
        if ARGS.infile:
            if ARGS.outfile:
                fingerprint_to_file(fingerprint, os.path.abspath(ARGS.outfile), ARGS.save_binary)
            else:
                print(fingerprint_to_string(fingerprint))
                s = ''
                for i, b in enumerate(fingerprint):
                    if i % 4 == 0:
                        print(s)  # always group 4 bytes
                        s = ''
                    s += '{:02x}'.format(b)
                print(s)
        elif ARGS.infolder:
            out_root = os.path.abspath(ARGS.outfolder)
            outfile = compute_outfile(ARGS.infolder, CURRENT_FILE, out_root, ARGS.save_binary)
            fingerprint_to_file(fingerprint, outfile, ARGS.save_binary)
    FINGERPRINT_IN_PROGRESS = False


def _read_and_resample(path):
    audio, sampling_rate = readWAV(path)
    resampled = resample(audio, sampling_rate, 8000, force_int16=True)  # Resample to 8 kHz
    return resampled


def on_spawned(spawn):
    print('on_spawn-added:', spawn)
    pending.append(spawn)
    event.set()


def pad_audio(audio_samples, target_seconds):
    """
    Pad the audio samples to at least the given number of seconds, but make sure that the resulting sample number
    is a multiple of 8192 (1024 ms).
    """
    samples_needed = target_seconds * 8192
    missing_samples = samples_needed - len(audio_samples)
    # Fill the audio with silence at the end
    if missing_samples > 0:
        audio_samples += [0] * missing_samples
    return audio_samples


def pad_to_8192_samples(audio_samples):
    """
    Pad the audio samples to a multiple of 8192 (1024 ms).
    """
    samples_needed = np.ceil(len(audio_samples) / 8192) * 8192
    missing_samples = int(samples_needed - len(audio_samples))
    if missing_samples > 0:
        # Fill audio with silence
        audio_samples += [0] * missing_samples
    return audio_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLI Batch fingerprinting tool for Zapr SDK contained in ABP Live News app"
                                                 "com.winit.starnews.hin")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--infile", metavar="FILE",
                       help="Read a single audio sample from path <FILE>. Only .wav format is supported.")
    group.add_argument("--infolder", metavar="INFOLDER",
                       help="Read all wav audio files from <FOLDER> and its subdirectories")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print", action="store_true",
                       help="Flag which indicates that the fingerprint will be written to STDOUT.")
    group.add_argument("--outfile", metavar="PATH", help="Write the fingerprint to path <PATH>.")
    group.add_argument("--outfolder", metavar="OUTFOLDER", help="Write the fingerprint to the folder <OUTFOLDER>."
                                                                "The directory structure is the same as <INFOLDER>.")

    parser.add_argument("--save_binary", action="store_true", help="Save the fingerprint directly to a binary file")
    parser.add_argument('--pad_seconds', type=int, help='If specified, the sound is padded with silence until it'
                                                        'reaches the given length in seconds, if necessary.')

    parser.add_argument("--algorithm", type=int, choices=[0, 1, 2, 3, 4], default=3,
                        help="Algorithm id that should be used for fingerprinting. Default is 3.")
    parser.add_argument("--log", action="store_true")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if (args.infile and args.outfolder) or (args.infolder and not args.outfolder):
        print("Invalid combination of infile/folder and outfile/folder")
        sys.exit(1)

    ARGS = args
    if args.pad_seconds:
        print(f"Padding audio to {args.pad_seconds} seconds.")
    print(f"Using algorithm {args.algorithm}")

    device = frida.get_usb_device()
    pending = []
    sessions = []
    scripts = []
    event = threading.Event()

    device.on('spawn-added', on_spawned)
    device.enable_spawn_gating()
    print('Enabled spawn gating')

    # Start pending spawns before starting script
    print('Pending:', device.enumerate_pending_spawn())

    for spawn in device.enumerate_pending_spawn():
        print('Resuming:', spawn)
        device.resume(spawn.pid)

    process_loaded = False
    session = None
    script = None

    while not process_loaded:
        # Do spawn gating until the target process (com.winit.starnews.hin:z_process) is finally loaded
        while len(pending) == 0:
            event.wait()
            event.clear()
        spawn = pending.pop()

        if spawn.identifier is not None and "com.winit.starnews.hin:z_process" in spawn.identifier:
            print('Instrumenting:', spawn)
            session = device.attach(spawn.pid)
            script = session.create_script(open('zapr/frida_zapr_injection.js').read())
            script.on('message', on_message)
            script.load()
            process_loaded = True
        device.resume(spawn.pid)

    if args.log:
        # Hook the logging functions and print everything.
        script.exports.init()

    script.exports.disablefingerprinting()
    if args.infile:
        complete_path = os.path.abspath(args.infile)
        CURRENT_FILE = complete_path
        audio_samples = _read_and_resample(complete_path)
        audio_samples = [int(x) for x in audio_samples]
        audio_samples = pad_to_8192_samples(audio_samples)
        if args.pad_seconds is not None:
            audio_samples = pad_audio(audio_samples, args.pad_seconds)
        script.exports.computeFingerprint(audio_samples, args.algorithm, args.log)
        input()
    elif args.infolder:
        infolder = os.path.abspath(args.infolder)
        wav_list = [y for x in os.walk(infolder) for y in glob(os.path.join(x[0], '*.wav'))]

        for i, file in enumerate(wav_list):
            if i > 0 and i % 100 == 0:
                logging.info(f"Finished {i}/{len(wav_list)} files")
            if os.path.getsize(file) >= 5536100:
                # Files larger than this size cause an error (rpcexception: script destroyed).
                continue
            while FINGERPRINT_IN_PROGRESS:
                # Ensure that the callback for the last file is finished (which sets the variable to false)
                time.sleep(0.1)

            FINGERPRINT_IN_PROGRESS = True
            CURRENT_FILE = file

            out_root = os.path.abspath(args.outfolder)
            outfile = compute_outfile(args.infolder, file, out_root, ARGS.save_binary)
            if os.path.exists(outfile):
                # Skip already fingerprinted files
                FINGERPRINT_IN_PROGRESS = False
                continue

            try:
                audio_samples = _read_and_resample(file)
                audio_samples = [int(x) for x in audio_samples]
            except ValueError:
                # This error occurs in at least one, but very few WAV files from Tidigits. Probably corrupted.
                logging.error("Error reading WAV. Continuing")
                FINGERPRINT_IN_PROGRESS = False
                continue
            # Always pad the audio to a multiple of 8192 with zero bytes
            audio_samples = pad_to_8192_samples(audio_samples)

            if args.pad_seconds is not None:
                audio_samples = pad_audio(audio_samples, args.pad_seconds)
            try:
                script.exports.computeFingerprint(audio_samples, args.algorithm, args.log)
            except frida.InvalidOperationError as ex:
                logging.error(f"Invalid operation error on file {file}: {str(ex)}")
                exit(1)
    print("Finished..")
