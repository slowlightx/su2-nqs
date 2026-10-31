"""Active training lift/JVP and sampler canonicalization preserve physics."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from su2_nqs.models.triangular_clebsch_active_layout import mixed_active_layout
from su2_nqs.models.triangular_active_train import lift_active_train_bank, canonical_active_sampling_bank
from su2_nqs.models.triangular_clebsch_feature_lut import prepare_feature_lut_bank, make_feature_lut_amplitude
from su2_nqs.su2.triangle_blocks import triangle_block_tables, nine_site_output


def test_actual_36_site_active_profile_lift_and_full_tangent_replay():
    cfg = MixedClebschConfig(6, 9, 4)
    profile = tuple({q: min(d, 1) for q, d in row['widths'].items()}
                    for row in mixed_clebsch_layout(cfg))
    rng = np.random.default_rng(2610279951)
    local = tuple(tuple(jnp.asarray(np.linalg.qr(rng.normal(size=(
        triangle_block_tables()['nine'][q].shape[1], nine_site_output(q, 9).width)))[0])
        for q in cfg.local_q) for _ in range(4))
    top = tuple(tuple(jnp.asarray(rng.normal(size=g['shape'])) for g in row['groups'])
                for row in mixed_active_layout(cfg, profile))
    bank = dict(local=local, top=top)
    delta = jax.tree.map(lambda x: jnp.asarray(rng.normal(size=x.shape))*.03, bank)
    row = jnp.asarray(rng.permutation(np.tile([0, 1], 18)))
    raw_active = make_feature_lut_amplitude(cfg, active_top_widths=profile)
    raw_native = make_feature_lut_amplitude(cfg)
    fa = lambda p: raw_active(prepare_feature_lut_bank(p, cfg), row)
    fn = lambda p: raw_native(prepare_feature_lut_bank(lift_active_train_bank(p, cfg, profile), cfg), row)
    va, da = jax.jvp(fa, (bank,), (delta,))
    vn, dn = jax.jvp(fn, (bank,), (delta,))
    assert max(abs(float(va)), abs(float(da))) > 0
    assert abs(va-vn) < 2e-11*max(abs(float(va)), 1e-300)
    assert abs(da-dn) < 2e-11*max(abs(float(da)), 1e-300)
    normalized, report = canonical_active_sampling_bank(bank, cfg, profile)
    assert abs(fa(normalized)*np.sqrt(report['original_raw_norm'])-va) < 2e-11*max(abs(float(va)), 1e-300)
    assert report['exact_inactive_parent_closure_verified']
