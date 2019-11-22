import os
import re


def has_ext(filename, ext):
    """
    Check if the filename ends in the extension

    Parameters
    ----------
    filename : str
        The name of the file
    ext : str
        The extension, may have leading dot (e.g txt == .txt)

    Returns
    -------
    bool indicating if the filename has the extension

    """
    if ext == None:
        return True
    if ext[0] != ".":
        ext = "." + ext
    return filename[-len(ext):].lower() == ext.lower()


def get_all_files_in_dir(
        in_dir, ext=None, return_absolute=True,
        recursive=False, verbose=False, re_filter=None):
    """
    Get all files in the directory with the given extension.

    Parameters
    ----------
    in_dir : str
        The absolute path to the directory
    ext : str, optional. Defaults to None.
        The extension of files to get.
    return_absolute : bool, optional. Defaults to True.
        Whether to return the absolute filename or not.
    recursive: bool, optional. Defaults to False.
        Whether to recurse through directories.
    verbose: bool, optional. Defaults to False.
        Whether to print the files found.

    Returns
    -------
    List : A list of filenames
    """
    if not os.path.isdir(in_dir):
        print("Non existant directory " + str(in_dir))
        return []

    def match_filter(f):
        if re_filter is None:
            return True
        search_res = re.search(re_filter, f)
        return search_res is not None

    def ok_file(root_dir, f):
        return (
            has_ext(f, ext) and match_filter(f) and
            os.path.isfile(os.path.join(root_dir, f)))

    def convert_to_path(root_dir, f):
        return os.path.join(root_dir, f) if return_absolute else f

    if verbose:
        print("Adding following files from {}".format(in_dir))

    if recursive:
        onlyfiles = []
        for root, _, filenames in os.walk(in_dir):
            start_root = root[:len(in_dir)]

            if len(root) == len(start_root):
                end_root = ""
            else:
                end_root = root[len(in_dir + os.sep):]
            for filename in filenames:
                filename = os.path.join(end_root, filename)
                if ok_file(start_root, filename):
                    to_add = convert_to_path(start_root, filename)
                    if verbose:
                        print(to_add)
                    onlyfiles.append(to_add)

    else:
        onlyfiles = [
            convert_to_path(in_dir, f) for f in sorted(os.listdir(in_dir))
            if ok_file(in_dir, f)
        ]
        if verbose:
            for f in onlyfiles:
                print(f)

    if verbose:
        print()
    return onlyfiles


def log_exception(ex, more_info=""):
    """
    Log an expection and additional info

    Parameters
    ----------
    ex : Exception
        The python exception that occured
    more_info : 
        Additional string to log

    Returns
    -------
    None

    """

    template = "{0} because exception of type {1} occurred. Arguments:\n{2!r}"
    message = template.format(more_info, type(ex).__name__, ex.args)
    print(message)
