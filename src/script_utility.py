import os
import time
import pickle
import logging
import tempfile
from pathlib import Path
from datetime import datetime as dt
from typing import Callable, Tuple, Dict


def set_logging(log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def print_runtime(func: Callable) -> Callable:
    """ Print the start time and the runtime of a script. """

    def wrapper():
        ss = dt.now().strftime('%d/%m/%y %H:%M:%S')
        print("----- Started {} -----\n".format(ss), flush=True)
        start_time = time.time()

        def print_end_time():
            es = dt.now().strftime('%H:%M:%S')
            elapsed = time.time() - start_time
            ll = ''.join(['-'] * 37)
            print("\n%s\n--- %s - %s ----\n--------- % 9.3f seconds ---------\n%s" % (ll, ss, es, elapsed, ll))

        try:
            func()
        except Exception as ex:
            print_end_time()
            raise ex
        print_end_time()

    return wrapper


def persist_to_file(original_func: Callable):
    """ Decorator to save function output to tmp directory (will be wiped on restart and needs to be re-created
     daily).
     """

    def wrapper(*args, **kws):
        now = dt.now().strftime('%Y%m%d')
        tmp_dir = Path(tempfile.gettempdir())

        arg_name = _get_arguments_as_string(args, kws)
        save_path = tmp_dir / ('{}_{}_{}.pickle'.format(original_func.__name__, now, arg_name))
        logging.info(f"Cache path `{save_path}`")

        try:
            cache = pickle.load(open(save_path, 'rb'))
        except (IOError, ValueError):
            cache = None

        if cache is None:
            cache = original_func(*args, **kws)
            pickle.dump(cache, open(save_path, 'wb'))

        return cache

    return wrapper


def persist_to_cache_folder(cached_folder: Path):
    """ Decorator to save function output to a specified folder. """

    def decorator(original_func: Callable):
        def wrapper(*args, **kws):
            arg_name = _get_arguments_as_string(args, kws)
            save_path = cached_folder / ('{}_{}.pickle'.format(original_func.__name__, arg_name))
            logging.info(f"Cache path `{save_path}`")

            try:
                cache = pickle.load(open(save_path, 'rb'))
            except (IOError, ValueError, pickle.UnpicklingError):
                if save_path.exists():
                    os.remove(save_path)
                cache = None

            if cache is None:
                cache = original_func(*args, **kws)
                pickle.dump(cache, open(save_path, 'wb'))

            return cache

        return wrapper

    return decorator


def _get_arguments_as_string(args: Tuple, kws: Dict) -> str:
    arg_name = ''
    for arg in args:
        if callable(arg):
            arg_name += '_' + str(arg.__name__)
        else:
            arg_name += '_' + str(arg)

    kws = dict(sorted(kws.items(), key=lambda item: item[0]))
    for name, val in kws.items():
        if callable(val):
            arg_name += '_' + '{}={}'.format(str(name), str(val.__name__))
        else:
            arg_name += '_' + '{}={}'.format(str(name), str(val))

    arg_name = str(arg_name).replace('/', '')
    arg_name = arg_name.strip('_')
    return arg_name
