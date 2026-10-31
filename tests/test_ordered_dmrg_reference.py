"""A coupling-changed MPS is explicitly a new physical reference run."""
import json
import pytest

tenpy = pytest.importorskip('tenpy')
from tenpy.networks.mps import MPS
from projects.clebsch_tree_nqs.workflows.spin_tensor_network_reference import (
    run, _checkpoint, file_sha256,
)
from su2_nqs.benchmarks.tensor_network_readout import tenpy_graph_model
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds


def test_warm_reference_retains_parent_bytes_and_enforces_hash(tmp_path):
    model = tenpy_graph_model(4, triangular_j1j2_bonds(2, 2, j2=.125, periodic=True))
    psi = MPS.from_singlets(model.lat.mps_sites()[0], 4, [(0, 1), (2, 3)], bc='finite')
    parent = tmp_path/'parent.h5'; _checkpoint(parent, psi, dict(j2=.125))
    digest = file_sha256(parent)
    spec = dict(output=str(tmp_path/'run'), rows=2, columns=2, j2=0.,
        initializations=['warm'], initial_checkpoints={'warm':str(parent)},
        initial_checkpoint_sha256={'warm':digest}, stages=[dict(chi=8,sweeps=1)],
        seconds_per_arm=120, safe_local_update_guard=True, reserve_seconds=60,
        memory_cap_gib=4, mixer=False, regions={'compact4':[0,1,2,3]})
    config = tmp_path/'spec.json'; config.write_text(json.dumps(spec))
    result = run(config, 'warm')
    assert result['status'] == 'bounded_acquisition_complete'
    assert result['history'][0]['sweeps'] == 1
    assert result['registration']['initial_checkpoint_sha256'] == digest
    assert result['registration']['config']['j2'] == 0
    assert file_sha256(parent) == digest
    assert not result['phase_claim']
    spec['output'] = str(tmp_path/'bad')
    spec['initial_checkpoint_sha256']['warm'] = '0'*64
    config.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='changed'):
        run(config, 'warm')
