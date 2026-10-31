"""One periodic triangle/nine-site cover and binary coarse-graining rule.

The family L=6*k has two fixed ternary layers, then a tile-preserving binary
top. Local frames follow physical120 coordinates and the repository's
row-major sites. Integer lifts, rather than modulo-six site copying, make
the same compact nine-site supports valid at every size.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral

from .geometry_tree import morton_order


_BASE_PATCHES = (
    ((0, 6, 7), (31, 1, 2), (29, 35, 30)),
    ((21, 27, 28), (26, 32, 33), (34, 4, 5)),
    ((3, 9, 10), (8, 14, 15), (16, 22, 23)),
    ((11, 17, 12), (18, 24, 25), (13, 19, 20)),
)
_NN = ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1))


@dataclass(frozen=True)
class TriangleNineSiteBlock:
    """Three CCW elementary triangles with a certified local C3 frame."""
    patches: tuple[tuple[int, int, int], ...]
    lifted_coordinates: tuple[tuple[int, int], ...]
    c3_translation: tuple[int, int]
    tile: tuple[int, int]
    base_index: int

    @property
    def sites(self) -> tuple[int, ...]:
        return tuple(s for patch in self.patches for s in patch)


@dataclass(frozen=True)
class TriangularSchurHierarchy:
    """Geometry only; the decoder must attach legal doubled-q intertwiners.

    Coarse IDs 0..B-1 are nine-site blocks. Each binary merge is
    (left,right,parent); parents are B,B+1,... in evaluation order. Local C3
    covariance does not make the entire tree globally A1.
    """
    length: int
    blocks: tuple[TriangleNineSiteBlock, ...]
    binary_merges: tuple[tuple[int, int, int], ...]

    @property
    def n_sites(self) -> int:
        return self.length**2

    @property
    def patches(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(patch for block in self.blocks for patch in block.patches)

    @property
    def layer_counts(self) -> tuple[int, ...]:
        sizes = [self.n_sites, self.n_sites//3, len(self.blocks)]
        heights = [0]*len(self.blocks)
        counts: dict[int, int] = {}
        for left, right, _ in self.binary_merges:
            height = 1+max(heights[left], heights[right])
            heights.append(height)
            counts[height] = counts.get(height, 0)+1
        for height in sorted(counts):
            sizes.append(sizes[-1]-counts[height])
        return tuple(sizes)

    @property
    def coarse_supports(self) -> tuple[tuple[int, ...], ...]:
        supports = [block.sites for block in self.blocks]
        for left, right, parent in self.binary_merges:
            if parent != len(supports):
                raise RuntimeError('noncanonical coarse-node ordering')
            supports.append(supports[left]+supports[right])
        return tuple(supports)


@lru_cache(maxsize=1)
def _base_integer_frames() -> tuple:
    output = []
    for patches in _BASE_PATCHES:
        support = {s for patch in patches for s in patch}
        start = min(support)
        lifts = {start: divmod(start, 6)}
        queue = [start]
        for site in queue:
            row, col = divmod(site, 6)
            for dr, dc in _NN:
                neighbor = ((row+dr) % 6)*6+(col+dc) % 6
                if neighbor not in support:
                    continue
                proposed = (lifts[site][0]+dr, lifts[site][1]+dc)
                if neighbor in lifts:
                    if lifts[neighbor] != proposed:
                        raise RuntimeError('base support has a winding cycle')
                else:
                    lifts[neighbor] = proposed
                    queue.append(neighbor)
        if set(lifts) != support:
            raise RuntimeError('base support is disconnected')
        translations = set()
        for port, patch in enumerate(patches):
            for i, site in enumerate(patch):
                row, col = lifts[site]
                rr, cc = lifts[patches[(port+1) % 3][(i+1) % 3]]
                translations.add((rr+col, cc-row+col))
        if len(translations) != 1:
            raise RuntimeError('base physical C3 frame is inconsistent')
        output.append((tuple(lifts[s] for patch in patches for s in patch),
                       next(iter(translations))))
    return tuple(output)


@lru_cache(maxsize=None)
def triangular_schur_hierarchy(length: int) -> TriangularSchurHierarchy:
    """Generate the fixed-bottom singlet family L=6*k for integer k>=1.

    Each Morton-ordered six-by-six tile first merges its four nine-site
    blocks by the original two binary levels. Above these 36-site modules,
    recursively bisect contiguous Morton intervals. Parents are numbered
    by height, then interval start. Thus every old power-of-two size has
    exactly its old ordered merges, while nonpowers avoid a large carried
    tail (L=18 has a 144|180-spin root, not 288|36). All nonroot intertwiners
    must retain legal q>0 sectors; a tile is not a frozen singlet state.
    """
    if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
        raise ValueError('length must be a positive integer')
    length = int(length)
    if length % 2:
        raise ValueError('odd N spin-half systems have no target_q=0 singlet')
    tiles = length//6
    if length % 6 or tiles < 1:
        raise ValueError('this fixed-bottom hierarchy requires L=6*k')
    blocks = []
    for tile_id in morton_order(tiles, tiles):
        tile_r, tile_c = divmod(tile_id, tiles)
        off_r, off_c = 6*tile_r, 6*tile_c
        for base_index, (coords, translation) in enumerate(_base_integer_frames()):
            lifted = tuple((r+off_r, c+off_c) for r, c in coords)
            sites = tuple((r % length)*length+c % length for r, c in lifted)
            patches = tuple(tuple(sites[3*i:3*i+3]) for i in range(3))
            # A(r,c)=(-c,r-c): t' = t + offset - A offset.
            t = (translation[0]+off_r+off_c,
                 translation[1]-off_r+2*off_c)
            blocks.append(TriangleNineSiteBlock(patches, lifted, t,
                                               (tile_r, tile_c), base_index))
    if sorted(s for block in blocks for s in block.sites) != list(range(length**2)):
        raise RuntimeError('periodic lifted cover is not an exact partition')
    merges = []
    active = list(range(len(blocks)))
    # Preserve the complete 6x6 module and its ordered child gauges first.
    for _ in range(2):
        following = []
        for i in range(0, len(active), 2):
            parent = len(blocks)+len(merges)
            merges.append((active[i], active[i+1], parent))
            following.append(parent)
        active = following
    intervals = []
    def bisect(start: int, stop: int) -> tuple[tuple[int, int], int]:
        key = (start, stop)
        if stop-start == 1:
            return key, 0
        middle = (start+stop)//2
        left, lh = bisect(start, middle)
        right, rh = bisect(middle, stop)
        height = 1+max(lh, rh)
        intervals.append((height, start, key, left, right))
        return key, height
    bisect(0, len(active))
    ids = {(i, i+1): parent for i, parent in enumerate(active)}
    for _, _, key, left, right in sorted(intervals):
        parent = len(blocks)+len(merges)
        merges.append((ids[left], ids[right], parent))
        ids[key] = parent
    return TriangularSchurHierarchy(length, tuple(blocks), tuple(merges))
