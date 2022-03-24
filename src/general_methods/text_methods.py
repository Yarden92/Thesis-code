import contextlib
import io
import sys

@contextlib.contextmanager
def silent_execute():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

