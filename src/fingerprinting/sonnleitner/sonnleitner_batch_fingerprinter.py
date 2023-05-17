import argparse
import os
import sys
import inspect
from multiprocessing import cpu_count, Pool

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from audio.io.audioReader import readWAV
from sonnleitner import Sonnleitner
from glob import glob

# Query settings that were determined for short queries using the sonnleitner benchmark.
# These settings still allow identification for short snippets.
# They can create fingerprints for ca. 98,5% of the speech commands v0.02 dataset
short_query_settings = {
    'max_filter_width': 71,
    'max_filter_height': 35,
    'quads_per_second': 500,
    'grouping_center': 122,  # 0.488 seconds in the future
    'grouping_width': 210  # window is 840ms wide [+12 to +232 frames, so +48 ms to +928ms]
}


def _compute_outfile(in_root, in_file, out_root):
    """
    The file_path is an abspath to a file somewhere under in_root. out_root is an abspath, as well.
    Returns the abspath to a file with same basename as file_path, but under out_root.
    The nesting of the infile and outfile (up to in_root/out_root) is the same.
    """
    sub_path = os.path.relpath(in_file, in_root)
    full_outpath = os.path.join(out_root, sub_path)
    full_outpath = full_outpath[:-4]  # remove .wav extension
    full_outpath += '.fprint'
    return full_outpath


def _fingerprint_to_string(fingerprint):
    """
    Create a printable string from the fingerprint by concatenating the quads (one per line).
    """
    res = "BEGIN\n"
    for quad in fingerprint.quads:
        coords = [f"{p.x},{p.y};" for p in [quad.A, quad.B, quad.C, quad.D]]
        res += ''.join(coords)
        res += '\n'
    res += '\n'
    for peak in fingerprint.peaks:
        res += f"{peak.x},{peak.y}\n"
    res += "END"
    return res


def _fingerprint_to_file(fingerprint, outfile):
    """
    Write a fingerprint object to a given file. Creates directories, if needed.
    """
    parent_folder = os.path.dirname(outfile)
    os.makedirs(parent_folder, exist_ok=True)
    with open(outfile, 'w') as f:
        outstring = _fingerprint_to_string(fingerprint)
        f.write(outstring)


def compute_and_save_fingerprint(params):
    """
    Create a fingerprint for a given file and write it to the outfile.
    Return True if no fingerprint is created. Else: False.
    """
    file, infolder, out_root = params
    outfile = _compute_outfile(infolder, file, out_root)
    if os.path.exists(outfile):
        return False, file
    try:
        wav, sampling_rate = readWAV(file)
    except ValueError:
        # Very few TIDIGITS wav files are malformed and reading them results in a ValueError.
        print(f"Error: Value error when reading {file}. No fingerprint generated")
        return True, file
    fingerprint = Sonnleitner.compute_fingerprint(wav, sampling_rate, custom_settings=short_query_settings)
    if not fingerprint:
        print(f"Error: No fingerprint was generated for {file}!")
        return True, file
    _fingerprint_to_file(fingerprint, outfile)
    return False, file


def main():
    parser = argparse.ArgumentParser(
        description="CLI Batch fingerprinting tool for Sonnleitner's algorithm")

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

    parser.add_argument("--n_threads", metavar='N', type=int, default=cpu_count(), choices=range(1, cpu_count()),
                        help="Number of threads to use (in folder processing). Default is using all CPU threads.")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if (args.infile and args.outfolder) or (args.infolder and not args.outfolder):
        print("Invalid combination of infile/folder and outfile/folder")
        sys.exit(1)

    if args.infile:
        # In this case, only a single file should be fingerprinted
        complete_path = os.path.abspath(args.infile)
        wav, sampling_rate = readWAV(complete_path)
        fingerprint = Sonnleitner.compute_fingerprint(wav, sampling_rate, custom_settings=short_query_settings)
        if not fingerprint:
            print(f"Error: No fingerprint was generated for {complete_path}!")
            return
        if args.outfile:
            _fingerprint_to_file(fingerprint, os.path.abspath(args.outfile))
        else:
            print(_fingerprint_to_string(fingerprint))
    elif args.infolder:
        # Read all wav files from the infolder and its subfolders.
        # Fingerprints are saved in the outfolder. The directory structure is mirrored
        infolder = os.path.abspath(args.infolder)
        out_root = os.path.abspath(args.outfolder)
        wav_list = [y for x in os.walk(infolder) for y in glob(os.path.join(x[0], '*.wav'))]

        with Pool(processes=args.n_threads) as p:
            results, files = zip(*p.map(compute_and_save_fingerprint, [(file, infolder, out_root) for file in wav_list]))
            n_errors = results.count(True)
            print(f"Encountered {n_errors} files for which no fingerprint could be generated")
        print('Errors on following files:')
        for res, file in zip(results, files):
            if res:
                print(file)


if __name__ == '__main__':
    main()
