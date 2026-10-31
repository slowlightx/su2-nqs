"""Check actual sites/bonds at several sizes, not only factor counts."""
import numpy as np
import pytest

from su2_nqs.lattices.triangular_schur_hierarchy import triangular_schur_hierarchy


@pytest.mark.parametrize('length,counts', [(6,(36,12,4,2,1)),
    (12,(144,48,16,8,4,2,1)), (18,(324,108,36,18,9,5,3,2,1)),
    (24,(576,192,64,32,16,8,4,2,1))])
def test_lifted_periodic_family_preserves_compact_support_and_frames(length, counts):
    h = triangular_schur_hierarchy(length)
    assert h.layer_counts == counts
    assert sorted(s for p in h.patches for s in p) == list(range(length**2))
    for block in h.blocks:
        coords = np.asarray(block.lifted_coordinates).reshape(3,3,2)
        np.testing.assert_array_equal(coords[:,1]-coords[:,0], np.tile([1,0],(3,1)))
        np.testing.assert_array_equal(coords[:,2]-coords[:,0], np.tile([1,1],(3,1)))
        tr, tc = block.c3_translation
        for port, patch in enumerate(block.patches):
            for local, site in enumerate(patch):
                r,c = divmod(site,length)
                image = ((-c+tr) % length)*length+(r-c+tc) % length
                assert image == block.patches[(port+1)%3][(local+1)%3]
        xy = set(map(tuple, coords.reshape(9,2)))
        steps = ((1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1))
        edges = sum((r+dr,c+dc) in xy for r,c in xy for dr,dc in steps)//2
        assert edges == 15
        diameter = 0
        for start in xy:
            distance = {start:0}; queue = [start]
            for r,c in queue:
                for dr,dc in steps:
                    other=(r+dr,c+dc)
                    if other in xy and other not in distance:
                        distance[other]=distance[(r,c)]+1;queue.append(other)
            assert len(distance)==9
            diameter = max(diameter, max(distance.values()))
        assert diameter == 3
    assert sorted(h.coarse_supports[-1]) == list(range(length**2))
    assert all(len(set(s))==len(s) for s in h.coarse_supports)


@pytest.mark.parametrize('length', [True, 0, -6, 9, 8, 18.0])
def test_impossible_singlet_or_unregistered_size_rule_is_rejected(length):
    with pytest.raises(ValueError):
        triangular_schur_hierarchy(length)


@pytest.mark.parametrize('length', [6, 12, 24, 48])
def test_original_power_of_two_merge_order_is_exactly_unchanged(length):
    tree = triangular_schur_hierarchy(length)
    active = list(range(len(tree.blocks))); original = []
    while len(active) > 1:
        following = []
        for left, right in zip(active[::2], active[1::2], strict=True):
            parent = len(tree.blocks)+len(original)
            original.append((left, right, parent)); following.append(parent)
        active = following
    assert tree.binary_merges == tuple(original)


def test_nonpower_tile_tree_is_near_balanced_and_preserves_complete_modules():
    tree = triangular_schur_hierarchy(18)
    supports = tree.coarse_supports
    assert len(tree.blocks) == 36 and len(tree.binary_merges) == 35
    assert tuple(len(supports[i]) for i in tree.binary_merges[-1][:2]) == (144, 180)
    assert [len(supports[p]) for _, _, p in tree.binary_merges[:18]] == [18]*18
    assert [len(supports[p]) for _, _, p in tree.binary_merges[18:27]] == [36]*9
    for left, right, parent in tree.binary_merges[27:]:
        assert abs(len(supports[left])-len(supports[right])) <= 36
        assert len(supports[parent]) == len(supports[left])+len(supports[right])
