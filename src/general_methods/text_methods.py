import contextlib
import io
import sys

@contextlib.contextmanager
def silent_execute():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


@staticmethod
def is_this_a_notebook() -> bool:
    return 'IPython' in sys.modules

