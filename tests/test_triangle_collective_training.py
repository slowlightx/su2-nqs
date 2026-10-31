"""Independent checks of the development-only collective training objective."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from projects.clebsch_tree_nqs.workflows.triangle_collective_training import projected_gain


def test_gain_matches_numpy_regularized_quadratic_and_rotations():
    rng = np.random.default_rng(74301)
    x = rng.normal(size=(31, 9))
    gram = x.T @ x/len(x)
    force = rng.normal(size=9)
    q = np.linalg.qr(rng.normal(size=(9, 2)))[0]
    sub, rhs = q.T @ gram @ q, q.T @ force
    inverse = np.linalg.inv(sub+.01*np.trace(sub)/2*np.eye(2))
    expected = rhs @ inverse @ rhs
    value = float(projected_gain(jnp.asarray(q), jnp.asarray(gram), jnp.asarray(force)))
    rotation = np.linalg.qr(rng.normal(size=(2, 2)))[0]
    rotated = float(projected_gain(jnp.asarray(q @ rotation), jnp.asarray(gram), jnp.asarray(force)))
    np.testing.assert_allclose(value, expected, atol=3e-15)
    np.testing.assert_allclose(rotated, expected, atol=3e-15)


def test_collective_training_gradient_matches_independent_finite_difference():
    rng = np.random.default_rng(74302)
    x = rng.normal(size=(24, 7))
    gram = jnp.asarray(x.T @ x/len(x))
    force = jnp.asarray(rng.normal(size=7))
    embedding = jnp.asarray(rng.normal(size=(7, 2)))
    direction = jnp.asarray(rng.normal(size=(7, 2)))
    def objective(y):
        q, _ = jnp.linalg.qr(y, mode="reduced")
        return projected_gain(q, gram, force)
    actual = float(jnp.sum(jax.grad(objective)(embedding)*direction))
    step = 2e-5
    expected = float((objective(embedding+step*direction)-objective(embedding-step*direction))/(2*step))
    np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)
