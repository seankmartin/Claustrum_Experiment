import subprocess
import os

from utils import get_all_files_in_dir, log_exception


def run_osf(args):
    subprocess.run(["osf", *args])


def run_capture_osf(args):
    result = subprocess.run(["osf", *args], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8')


def get_help():
    run_osf(["-h"])


def remove_file(remote):
    run_osf(["remove", remote])


def upload_file(local, remote):
    run_osf(["upload", local, remote])


def list_files():
    run_osf(["ls"])


def get_osf_files():
    result = run_capture_osf(["ls"])
    return [
        line[len("osfstorage\\"):].replace("/", "\\")
        for line in result.splitlines()]


def empty_osf():
    for file in get_osf_files():
        print("Removing {}".format(file))
        remove_file(file)


def upload_folder(folder, recursive=True):
    file_list = get_all_files_in_dir(folder, recursive=recursive)
    remote_list = [fname[len(folder + os.sep):] for fname in file_list]
    current_remote = get_osf_files()
    print("Beginning upload process")
    with open(os.path.join(folder, "uploaded_files.txt"), "w") as f:
        for local, remote in zip(file_list, remote_list):
            if not remote in current_remote:
                print(remote)
                print(current_remote)
                exit(-1)
                upload_file(local, remote)
                f.write("Uploaded {} to {}\n".format(
                    local, remote))
                print("Uploaded {} to {}".format(
                    local, remote))
            else:
                f.write("Skipped upload of {} to {} - already in OSF\n".format(
                    local, remote))
                print("Skipped upload of {} to {} - already in OSF".format(
                    local, remote))


def main(location):
    upload_folder(location)


if __name__ == "__main__":
    # NOTE please change this to be your password and change .osfcli.config
    your_osf_password = "Can't Steal This"
    os.environ["OSF_PASSWORD"] = your_osf_password
    location = r"C:\Users\smartin5\Neuroscience\Misc"
    main(location)
