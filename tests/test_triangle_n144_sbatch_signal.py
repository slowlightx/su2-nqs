"""Exercise the actual shell helper with real signalled mock children.

No Slurm/GPU or wavefunction is involved. These are execution-safety checks,
not proofs that native JAX kernels can stop before an external hard kill.
"""
from pathlib import Path
import os
import shlex
import signal
import subprocess
import sys
import time

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/remote/triangle_n144_signal_safe_20261018.sbatch'


def _driver(tmp_path, *, between_phases=False):
    source = SCRIPT.read_text().split('# SIGNAL_SAFE_CHILD_BEGIN\n', 1)[1].split(
        '# SIGNAL_SAFE_CHILD_END', 1)[0]
    child = tmp_path/'child.py'
    child.write_text('''import pathlib,signal,sys,time
p=pathlib.Path(sys.argv[1]);phase=sys.argv[2];stop=[]
signal.signal(signal.SIGUSR1,lambda *_:stop.append(True))
(p/(phase+'.ready')).write_text('ready')
if phase=='normal':
 (p/(phase+'.done')).write_text('done');sys.exit(0)
while not stop:time.sleep(.01)
time.sleep(.1)
(p/(phase+'.checkpoint')).write_text('checkpoint complete')
''')
    command = shlex.join([sys.executable, str(child), str(tmp_path)])
    calls = ('run_child '+command+' normal\nkill -USR1 $$\n' if between_phases
        else 'run_child '+command+' first\n')
    driver = tmp_path/'driver.sh'
    driver.write_text('set -euo pipefail\nNQS_CHILD_PID=""\nNQS_STOP_REQUESTED=0\n'+source+
        '\n'+calls+'run_child '+command+' second\n')
    return driver


@pytest.mark.parametrize('requested_signal', [signal.SIGUSR1, signal.SIGTERM])
def test_shell_forwards_stop_waits_for_checkpoint_and_does_not_start_replay(tmp_path, requested_signal):
    driver = _driver(tmp_path)
    process = subprocess.Popen(['bash', str(driver)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not (tmp_path/'first.ready').exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                break
            time.sleep(.01)
        assert (tmp_path/'first.ready').exists()
        os.kill(process.pid, requested_signal)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        assert (tmp_path/'first.checkpoint').read_text() == 'checkpoint complete'
        assert not (tmp_path/'second.ready').exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_stop_between_phases_does_not_start_replay(tmp_path):
    driver = _driver(tmp_path, between_phases=True)
    done = subprocess.run(['bash', str(driver)], capture_output=True, text=True, timeout=5)
    assert done.returncode == 0, (done.stdout, done.stderr)
    assert (tmp_path/'normal.done').exists()
    assert not (tmp_path/'second.ready').exists()
