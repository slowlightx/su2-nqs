"""Independent tiny-H oracle for the opt-in fixed physical-seed anchor."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.explorations.ct_protected_triplet_update_20261029 import anchored_amplitude


def test_anchor_preserves_source_and_all_native_tangents_and_exact_H():
    source=jnp.array([.4+.2j,.7-.1j]); reference=jnp.array([.3,.6])
    matrix=jnp.array([[.2+.3j,.7],[.5,.1-.4j]])
    model=lambda p:matrix@p
    template=jnp.array([.8,.2]); ref=model(template);scale=3.2
    amplitude=lambda p:anchored_amplitude(source,model(p),ref,scale)
    np.testing.assert_array_equal(amplitude(template),source)
    direction=jnp.array([.1,.4])
    _,tangent=jax.jvp(amplitude,(template,),(direction,))
    np.testing.assert_allclose(tangent,scale*matrix@direction,atol=1e-14)
    h=jnp.array([[.3,-.8],[-.8,-.2]])
    new=template+direction
    np.testing.assert_allclose(h@amplitude(new),
        anchored_amplitude(h@source,h@model(new),h@ref,scale),atol=1e-14)
