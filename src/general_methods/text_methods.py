import contextlib
from dataclasses import dataclass
import io
import sys


@contextlib.contextmanager
def silent_execute():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


def is_this_a_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

@dataclass
class FileNames:
    x: str = 'data_x.npy'
    y: str = 'data_y.npy'
    conf_json: str = 'conf.json'
    conf_yml: str = 'conf.yml'