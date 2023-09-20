import sys
import os, os.path as osp 
import shutil
import traceback

import pickle 
from typing import Any, IO

def save_pickle(fpath: str, data: Any) :
    """Save data in pickle format."""
    with open(fpath, 'wb+') as f :
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL);


def load_pickle(fpath: str) -> Any :
    """Load data from pickle file."""
    with open(fpath, 'rb') as f :
        data = pickle.load(f);
    return data;


def get_file_handle(fpath: str, mode: str) -> IO :
    try :
        fhand = open(fpath, mode);
    except Exception :
        traceback.print_exc();
        sys.exit();

    return fhand;


def get_nlines_in_file(fpath: str) -> int :
    assert osp.isfile(fpath), f"File not found = {fpath}";
    count = 0;
    fhand = get_file_handle(fpath, 'r');
    for line in fhand :
        line = line.strip();
        if len(line) > 0 :
            count += 1;
    fhand.close();

    return count;


def print_argparser_args(args) :
    """prints the argparser args better"""
    for arg in vars(args):
        print(arg, '=', getattr(args, arg));    


def get_free_port() :
    import socket
    from contextlib import closing    

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])   


def mkdir_rm_if_exists(dir_) :
    if osp.isdir(dir_) :
        shutil.rmtree(dir_);
    os.makedirs(dir_);