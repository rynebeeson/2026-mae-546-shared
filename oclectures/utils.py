"""Small notebook utilities."""
import io
import os
import tempfile
from contextlib import contextmanager, redirect_stdout


def redirect_outputs_to_tmp():
    """Send OpenMDAO's per-problem output folders (``problemN_out/`` with
    coloring caches, recorder databases, ...) to a session temp directory
    instead of the notebook folder. Call once, before building problems."""
    os.environ['OPENMDAO_WORKDIR'] = tempfile.mkdtemp(prefix='om_out_')


@contextmanager
def quiet():
    """Suppress stdout inside the block (e.g. dymos/OpenMDAO setup chatter),
    and remove the setup-check log file (``openmdao_checks.out``) that
    ``phase.simulate()`` unavoidably writes into the working directory."""
    existed = os.path.exists('openmdao_checks.out')
    try:
        with redirect_stdout(io.StringIO()):
            yield
    finally:
        if not existed and os.path.exists('openmdao_checks.out'):
            os.remove('openmdao_checks.out')
