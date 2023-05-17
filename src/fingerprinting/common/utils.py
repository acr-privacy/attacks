import base64
import os


def compute_outfile(in_root, in_file, out_root, save_binary=False):
    """
    The file_path is an abspath to a file somewhere under in_root. out_root is an abspath, as well.
    Returns the abspath to a file with same basename as file_path, but under out_root.
    The nesting of the infile and outfile (up to in_root/out_root) is the same.
    If save_binary is set, the file extension is '.fprint_bin'
    """
    sub_path = os.path.relpath(in_file, in_root)
    full_outpath = os.path.join(out_root, sub_path)
    full_outpath = full_outpath[:-4]  # remove .wav extension
    if save_binary:
        full_outpath += '.fprint_bin'
    else:
        full_outpath += '.fprint'
    return full_outpath


def fingerprint_to_string(fingerprint):
    """
    Create a printable string from the byte fingerprint by using base64 encoding.
    """
    res = "BEGIN\n"
    res += base64.b64encode(fingerprint).decode('utf-8')
    res += "\nEND"
    return res


def fingerprint_to_file(fingerprint, outfile, save_binary=False):
    """
    Write a fingerprint object to a given file. Creates directories, if needed.
    """
    parent_folder = os.path.dirname(outfile)
    os.makedirs(parent_folder, exist_ok=True)
    if save_binary:
        with open(outfile, 'wb') as f:
            f.write(fingerprint)
    else:
        with open(outfile, 'w') as f:
            outstring = fingerprint_to_string(fingerprint)
            f.write(outstring)
