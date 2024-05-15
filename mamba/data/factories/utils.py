import os
import urllib.request


def check_filesize(target, url):
    try:
        filesize = os.path.getsize(target)
    except FileNotFoundError:
        return False
    target_filesize = urllib.request.urlopen(url).length
    return filesize == target_filesize


def touch(path):
    with open(path, "w") as f:
        f.write("")
