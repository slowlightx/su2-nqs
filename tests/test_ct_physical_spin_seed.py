"""The supervised projector objective is not mistaken for physical energy."""
import numpy as np
from projects.clebsch_tree_nqs.explorations.ct_physical_spin_seed_20261029 import fidelity,projector_training_action


def test_complex_projector_loss_gradient_and_gauge():
    rng=np.random.default_rng(29139)
    p=rng.normal(size=9)+1j*rng.normal(size=9)
    t=rng.normal(size=9)+1j*rng.normal(size=9)
    d=rng.normal(size=9)+1j*rng.normal(size=9);Q=rng.random(9)+.1
    hp=projector_training_action(p,t,Q)
    E=np.vdot(p,hp/Q).real/np.sum(abs(p)**2/Q)
    assert abs(E-(1-fidelity(p,t,Q)))<5e-16
    gradient=2*np.vdot(d,(hp-E*p)/Q).real/np.sum(abs(p)**2/Q)
    eps=1e-6
    exact=((1-fidelity(p+eps*d,t,Q))-(1-fidelity(p-eps*d,t,Q)))/(2*eps)
    assert abs(exact-gradient)<2e-10
    assert abs(fidelity(p,t,Q)-fidelity((3+2j)*p,(-2+.7j)*t,Q))<5e-16
    np.testing.assert_allclose(projector_training_action(t,t,Q),0.,atol=2e-15)
