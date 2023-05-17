import os, sys, glob
import subprocess

speaker_dataset = "../../data/processed/"

sonnleitner_location = "../../data/fingerprints/sonnleitner/"
acrcloud_location    = "../../data/fingerprints/acrcloud/"
zapr0_location       = "../../data/fingerprints/zapr0/"
zapr3_location       = "../../data/fingerprints/zapr3/"

fingerprint_paths = [sonnleitner_location, acrcloud_location, zapr0_location, zapr3_location]


def setup():
    """Make sure all the necessary folders to save the fingerprints in exist."""

    if not os.path.exists(sonnleitner_location):
        os.makedirs(sonnleitner_location)
    if not os.path.exists(acrcloud_location):
        os.makedirs(acrcloud_location)
    if not os.path.exists(zapr0_location):
        os.makedirs(zapr0_location)
    if not os.path.exists(zapr3_location):
        os.makedirs(zapr3_location)

    # Check if there is any speaker dataset in the processed folder (ignore hidden files).
    if [f for f in os.listdir(speaker_dataset) if not f.startswith(".")] == []:
        print("~~ ERROR: Could not generate fingerprints with empty folder.")
        print("          Please save the speaker dataset in ../../data/processed")
        exit(1)

    print("\nMake sure the phone is connected via USB and adb is running.")
    print("Run the frida-server using for example the following commands:")
    print(' $ adb shell "su -c chmod 755 /data/local/tmp/frida-server"')
    print(' $ adb shell "su -c /data/local/tmp/frida-server &"\n')


def generate_fingerprints():
    """Generate all the fingerprints for different algorithms in subprocesses."""

    print(">> Generating the Sonnleitner fingerprints ...\n")
    subprocess.run(
        [
            "python3",
            "sonnleitner/sonnleitner_batch_fingerprinter.py",
            "--infolder",
            speaker_dataset,
            "--outfolder",
            sonnleitner_location,
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    print("\n>> Generating the ACRcloud fingerprints ...\n")
    subprocess.run(
        [
            "python3",
            "acrcloud/batch_fingerprint_deezer.py",
            "--infolder",
            speaker_dataset,
            "--outfolder",
            acrcloud_location,
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    print("\n!! Please close and open the com.winit.starnews.hin app now.")
    print("!! Additionally, the phone needs to be connected to the internet now.")
    print("\n>> Generating the Zapr fingerprints using algorithm 0 ...\n")
    subprocess.run(
        [
            "python3",
            "zapr/batch_fingerprint_zapr.py",
            "--infolder",
            speaker_dataset,
            "--outfolder",
            zapr0_location,
            "--algorithm",
            "0",
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

    print("\n!! Please close and open the com.winit.starnews.hin app now.")
    print("!! Additionally, the phone needs to be connected to the internet now.")
    print("\n>> Generating the Zapr fingerprints using algorithm 3 ...\n")
    subprocess.run(
        [
            "python3",
            "zapr/batch_fingerprint_zapr.py",
            "--infolder",
            speaker_dataset,
            "--outfolder",
            zapr3_location,
        ],
        stderr=subprocess.STDOUT,
        check=True,
    )

def balance_dataset():
    """This function makes sure, every speaker has the same amount of (400) fingerprints.
        Otherwise, the machine learning process might learn some voices better than others.
    """

    print("\n>> Balancing the data to make sure every speaker has exactly 400 fingerprints.")

    # Make sure every algorithm in the fingerprints dataset is checked.
    for path in fingerprint_paths:

        # Find out every speaker (here we assume 10 speakers).
        for speaker in range(1, 11):

            # Determine all files for a certain speaker and make sure single-digit numbers are formatted.
            files = glob.glob(path + str(speaker).zfill(2) + "*.fprint", recursive = True)

            # Reverse the list of files to delete the only the latest fingerprints.
            reversed_files = files[::-1]
            for i in range(len(files) - 400):
                os.remove(reversed_files[i])


def main():
    print("+---------------------------------------+")
    print("| SEMI-AUTOMATIC FINGERPRINT GENERATION |")
    print("+---------------------------------------+")

    setup()
    generate_fingerprints()
    balance_dataset()

    print("\nDone!")


if __name__ == "__main__":
    main()
