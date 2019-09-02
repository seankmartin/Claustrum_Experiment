import os
from functools import partial
from shutil import copyfile


def merge_2eegs(
        eeg1_location, eeg2_location,
        output_location=None, test_difference=False):
    if output_location is None:
        base1 = os.path.basename(eeg1_location)
        base2 = os.path.basename(eeg2_location)
        output_location = (
            eeg1_location.split(".")[0] + "_MERGE_" +
            base2.split(".")[0] + "." + base1.split(".")[1])
        print("Saving to " + output_location)

    with open(eeg1_location, 'rb') as f1, \
            open(eeg2_location, 'rb') as f2, \
            open(output_location, 'wb') as target_f:

        while True:
            line = f2.readline()
            try:
                line = line.decode('UTF-8')
            except:
                break

            if line == '':
                break
            if line.startswith('num_EEG_samples'):
                f2_samples = (int(''.join(line.split()[1:])))
                break

        f2.seek(0, 0)
        while True:
            line = f1.readline()
            try:
                line = line.decode('UTF-8')
            except:
                break

            if line == '':
                break
            if line.startswith('num_EEG_samples'):
                f1_samples = (int(''.join(line.split()[1:])))
                line = ('num_EEG_samples ' + str(f1_samples + f2_samples) +
                        "    \r\n")
            target_f.write(line.encode('UTF-8'))

        f1.seek(0, 0)
        while True:
            try:
                buff = f1.read(10).decode('UTF-8')
            except:
                break
            if buff == 'data_start':
                # header_offset = f1.tell()
                target_f.write(buff.encode('UTF-8'))
                break
            else:
                f1.seek(-9, 1)

        for _bytes in iter(partial(f1.read, 1024), b''):
            test = _bytes[-12:]
            try:
                val = test.decode('UTF-8')
                if val.startswith("\r\ndata_end"):
                    target_f.write(_bytes[:-12])
                else:
                    target_f.write(_bytes)
            except:
                target_f.write(_bytes)

        while True:
            try:
                buff = f2.read(10).decode('UTF-8')
            except:
                break
            if buff == 'data_start':
                # header_offset = f2.tell()
                break
            else:
                f2.seek(-9, 1)

        for _bytes in iter(partial(f2.read, 1024), b''):
            target_f.write(_bytes)

    if test_difference:
        with open(eeg1_location, 'rb') as f1, \
                open(eeg2_location, 'rb') as f2, \
                open(output_location, 'rb') as f3:
            byte1 = f1.read(1)
            byte3 = f3.read(1)
            i = 0
            while byte1:
                if byte1 != byte3:
                    print(i, byte1, byte3)
                    if i < 1000:
                        i = i + 1
                        byte3 = f3.read(1)
                    byte1 = f1.read(1)
                else:
                    i = i + 1
                    byte1 = f1.read(1)
                    byte3 = f3.read(1)

            while True:
                try:
                    buff = f2.read(10).decode('UTF-8')
                except:
                    break
                if buff == 'data_start':
                    # header_offset = f2.tell()
                    break
                else:
                    f2.seek(-9, 1)

            byte2 = f2.read(1)

            while byte2:
                if byte2 != byte3:
                    print(i, byte2, byte3)
                i = i + 1
                byte3 = f3.read(1)
                byte2 = f2.read(1)

    return output_location


def main(args):
    for i in range(32):
        a_val = ""
        if i is not 0:
            a_val = str(i + 1)
        output_location = merge_2eegs(
            args["eeg1_location"] + a_val, args["eeg2_location"] + a_val)

    src = args["eeg1_location"].split(".")[0] + ".set"
    dst = output_location.split(".")[0] + ".set"
    print("Copied set file to " + dst)
    copyfile(src, dst)
    test_merge(output_location)


def test_merge(loc):
    from neurochat.nc_lfp import NLfp
    lfp = NLfp()
    lfp.set_filename(loc)
    lfp.load()
    print(lfp.get_total_samples())
    print(lfp.get_duration())


if __name__ == "__main__":
    root = r"C:\Users\smartin5\Recordings\ER\29082019-bt2"
    args = {
        "eeg1_location": os.path.join(
            root, "29082019-bt2-1st-LFP1.eeg"),
        "eeg2_location": os.path.join(
            root, "29082019-bt2-1st-LFP2.eeg")
    }
    main(args)
