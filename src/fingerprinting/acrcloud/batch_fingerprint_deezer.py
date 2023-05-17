"""
Python script to batch fingerprint wav files using the ACRCloud implementation in Deezer Android App.
The target Deezer version is 6.1.14.99, however with small changes, a recent version 6.2.13.151 can be attacked as well.
The script uses the corresponding Frida injection script 'frida_injection.js'
"""

import argparse
import frida
import logging
import os
import sys
import time

from glob import glob

sys.path.append(os.getcwd())

from audio.utils import resample
from audio.io.audioReader import readWAV
from common.utils import fingerprint_to_file, fingerprint_to_string, compute_outfile


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


def _bytes_from_16bit_pcm(samples):
    """
    Given a list of 16 bit PCM samples, we construct a list of bytes (each 16 bit in little endian).
    """
    res_bytes = []
    for b in samples:
        # Compute the 2-complement, little endian of each 16-bit integer
        b1, b2 = int(b).to_bytes(2, byteorder='little', signed=True)
        res_bytes += [b1, b2]
    return res_bytes


if __name__ == '__main__':
    """
    Create fingerprints of one wav file (or multiple ones / a whole folder) using a Frida injection script.
    """
    parser = argparse.ArgumentParser(description="CLI Batch fingerprinting tool for Deezer Android app v. 6.1.14.99")

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
                                                        'reaches the given length in seconds, if necessary. '
                                                        'Typically, Deezer needs a length of at least 2 seconds.')

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

    device = frida.get_usb_device()
    pid = device.spawn(["deezer.android.app"])  # For Deezer 6.2.13.151, the package name is deezer.android.app
    device.resume(pid)
    time.sleep(1)  # Without it Java.perform silently fails
    session = device.attach(pid)
    script = session.create_script(open('acrcloud/frida_deezer_injection.js').read())
    script.load()

    script.on('message', on_message)

    if args.infile:
        complete_path = os.path.abspath(args.infile)
        CURRENT_FILE = complete_path
        audio_samples = _read_and_resample(complete_path)
        audio_bytes = _bytes_from_16bit_pcm(audio_samples)
        if args.pad_seconds:
            bytes_needed = args.pad_seconds * 16000  # one second of audio consists of 8000 2-byte samples
            missing_bytes = bytes_needed - len(audio_bytes)
            # Fill the audio with silence at the end
            if missing_bytes > 0:
                audio_bytes += [0] * missing_bytes

        script.post({'type': 'wave', 'payload': audio_bytes})
        input()
    elif args.infolder:
        infolder = os.path.abspath(args.infolder)
        wav_list = [y for x in os.walk(infolder) for y in glob(os.path.join(x[0], '*.wav'))]

        for i, file in enumerate(wav_list):
            if i > 0 and i % 100 == 0:
                logging.info(f"Finished {i} files")

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
            except ValueError:
                # This error occurs in at least one, but very few WAV files from Tidigits. Probably corrupted.
                logging.error("Error reading WAV. Continuing")
                FINGERPRINT_IN_PROGRESS = False
                continue
            audio_bytes = _bytes_from_16bit_pcm(audio_samples)

            if args.pad_seconds:
                bytes_needed = args.pad_seconds * 16000  # one second of audio consists of 8000 2-byte samples
                missing_bytes = bytes_needed - len(audio_bytes)
                # Fill the audio with silence at the end
                if missing_bytes > 0:
                    audio_bytes += [0] * missing_bytes
            script.post({'type': 'wave', 'payload': audio_bytes})

    # Signalize the end to the Frida script
    script.post({'type': 'wave', 'payload': None})
    print("Finished..")
    sys.exit(0)
