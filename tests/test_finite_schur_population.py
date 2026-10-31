"""Fixed force amendments, explicit data reuse and a separate spectral policy."""
import json
from pathlib import Path
import ast
import pytest
from projects.clebsch_tree_nqs.workflows.spin_finite_schur_population import CONFIG,registration,known_blocks


def test_population_force_protocol_preserves_parent_and_budgets():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    blocks=known_blocks(spec)
    assert [len(b['q']) for b in blocks]==[64,128,128]
    assert all(set(b)=={'states','q','carrier','hcarrier','groups'} for b in blocks)
    assert spec['steps']==1 and spec['protection']['maximum_variance_ratio']==1.
    assert not any('audit.npz' in k or 'candidate' in k for k in reg['inputs'])
    assert 'spectral_control' in spec and '5percent' in spec['spectral_control']
    for change in ({'maximum_q':2},{'relative_cutoff':.001},{'steps':8},
                   {'H2_terminal_capacities':[16384]},
                   {'seeds':{**spec['seeds'],'calibration':2026100291}}):
        with pytest.raises(ValueError):registration({**spec,**change})


def test_geometry_serialization_does_not_duplicate_reserved_force_field():
    import projects.clebsch_tree_nqs.workflows.spin_finite_schur_population as workflow
    tree=ast.parse(Path(workflow.__file__).read_text())
    geometry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='geometry')
    calls=[n for n in ast.walk(geometry) if isinstance(n,ast.Call) and
        isinstance(n.func,ast.Attribute) and n.func.attr=='savez_compressed']
    assert len(calls)==1
    assert 'hermitian_gradients' not in [kw.arg for kw in calls[0].keywords]
