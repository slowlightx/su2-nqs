"""Independent physical permutation, Pauli, spinor, and local-H C3 oracles."""
from functools import lru_cache
from itertools import product

import numpy as np
import pytest

from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.quantum_numbers import magnetic_numbers
from su2_nqs.su2.triangle_c3 import (
    C3Multiplicity, project_ternary_synthesis, ternary_c3_representation,
    ternary_coupled_basis, ternary_cycle_block, ternary_intertwiner_basis,
    triangle_chiral_schur,
)


def _physical_image(n,image):
    result=np.zeros(2**n,dtype=int)
    for packed in range(2**n):
        for old,new in enumerate(image):
            result[packed]|=((packed>>(n-old-1))&1)<<(n-new-1)
    return result


def _permuted(columns,image):
    return columns[np.argsort(image)]


def _spin_half_operators(n):
    local=[np.array([[0,.5],[.5,0]],complex),np.array([[0,.5j],[-.5j,0]],complex),
           np.diag([-.5,.5]).astype(complex)]
    operators=[]
    for site in range(n):
        axes=[]
        for op in local:
            matrix=np.array([[1.]],complex)
            for i in range(n):
                matrix=np.kron(matrix,op if i==site else np.eye(2))
            axes.append(matrix)
        operators.append(axes)
    return operators


def test_triangle_cycle_chirality_reflection_and_time_reversal_are_physical():
    u,labels=triangle_chiral_schur()
    np.testing.assert_allclose(u.conj().T @ u,np.eye(8),atol=8e-16,rtol=0)
    cycle=_physical_image(3,[1,2,0]);mirror=_physical_image(3,[1,0,2])
    omega=np.exp(2j*np.pi/3)
    np.testing.assert_allclose(_permuted(u,cycle),u*np.asarray([omega**ell for _,_,ell in labels]),atol=1e-15,rtol=0)
    positions={label:i for i,label in enumerate(labels)}
    for i,(q,m,ell) in enumerate(labels):
        np.testing.assert_allclose(_permuted(u[:,i],mirror),u[:,positions[q,m,-ell]],atol=1e-15,rtol=0)
        phase=np.asarray([(-1.)**p.bit_count() for p in range(8)])
        time_reversed=phase*u[np.arange(8)^7,i].conj()
        np.testing.assert_allclose(time_reversed,(-1.)**((q-m)//2)*u[:,positions[q,-m,-ell]],atol=1e-15,rtol=0)
    spins=_spin_half_operators(3)
    chirality=np.zeros((8,8),complex)
    for a,b,c,sign in [(0,1,2,1),(1,2,0,1),(2,0,1,1),(0,2,1,-1),(2,1,0,-1),(1,0,2,-1)]:
        chirality+=sign*(spins[0][a] @ spins[1][b] @ spins[2][c])
    expected=[-np.sqrt(3)/4*ell if q==1 else 0. for q,_,ell in labels]
    np.testing.assert_allclose(u.conj().T @ chirality @ u,np.diag(expected),atol=8e-16,rtol=0)
    # This multiplicity is physical bond resonance, not another spin-half.
    # Its xy plane selects the three bonds; z is scalar, T-odd chirality.
    doublet=u[:,[positions[1,1,ell] for ell in (1,-1)]]
    tau_x=np.array([[0,1],[1,0]],complex);tau_y=np.array([[0,-1j],[1j,0]],complex)
    for (first,second),angle in zip(((0,1),(1,2),(2,0)),(0,2*np.pi/3,4*np.pi/3),strict=True):
        bond=sum(spins[first][axis] @ spins[second][axis] for axis in range(3))
        reduced=-np.eye(2)/4+(np.cos(angle)*tau_x+np.sin(angle)*tau_y)/2
        np.testing.assert_allclose(doublet.conj().T @ bond @ doublet,reduced,atol=5e-16,rtol=0)
    # Scalar chirality commutes with spin rotations; vector chirality does not.
    for axis in range(3):
        total=sum(site[axis] for site in spins)
        np.testing.assert_allclose(chirality @ total-total @ chirality,0,atol=5e-16,rtol=0)
    vector_z=sum(spins[i][0] @ spins[j][1]-spins[i][1] @ spins[j][0] for i,j in [(0,1),(1,2),(2,0)])
    vector_components=[]
    for axis,(first,second) in enumerate(((1,2),(2,0),(0,1))):
        vector=sum(spins[i][first] @ spins[j][second]-spins[i][second] @ spins[j][first]
                   for i,j in ((0,1),(1,2),(2,0)))
        vector_components.append(vector)
        total=sum(site[axis] for site in spins)
        # Exact full-eight-dimensional bridge, not just a projected analogy:
        # physical vector chirality = 4 * scalar chirality * physical spin.
        np.testing.assert_allclose(vector,4*chirality @ total,atol=5e-16,rtol=0)
    doublets=u[:,[i for i,(q,_,_) in enumerate(labels) if q==1]]
    p_doublet=doublets @ doublets.conj().T
    np.testing.assert_allclose(chirality @ chirality,3*p_doublet/16,atol=5e-16,rtol=0)
    strength=sum(component @ component for component in vector_components)
    np.testing.assert_allclose(strength,9*p_doublet/4,atol=8e-16,rtol=0)
    bond_sum=sum(spins[i][axis] @ spins[j][axis] for i,j in ((0,1),(1,2),(2,0))
                 for axis in range(3))
    # Local strength is wholly fixed by two-body data, not a new QSL signal.
    np.testing.assert_allclose(strength,9*np.eye(8)/8-3*bond_sum/2,atol=5e-16,rtol=0)
    sx=sum(site[0] for site in spins)
    assert np.linalg.norm(vector_z @ sx-sx @ vector_z)>1
    t=np.array([[0,1],[-1,0]],complex)
    full_t=np.kron(np.kron(t,t),t)
    np.testing.assert_allclose(full_t @ chirality.conj() @ full_t.conj().T,-chirality,atol=5e-16,rtol=0)
    np.testing.assert_allclose(full_t @ vector_z.conj() @ full_t.conj().T,vector_z,atol=5e-16,rtol=0)


@pytest.mark.parametrize('charges,total_q', [((1,1,1),1),((1,1,1),3),((1,3,1),3),
                                          ((3,1,3),3),((0,2,2),2)])
def test_two_F_moves_and_exchange_match_independent_product_space_cycle(charges,total_q):
    q0,q1,q2=charges
    block,source,target=ternary_cycle_block(q0,q1,q2,total_q)
    for m in magnetic_numbers(total_q):
        old,old_channels=ternary_coupled_basis(q0,q1,q2,total_q,m)
        new,new_channels=ternary_coupled_basis(q2,q0,q1,total_q,m)
        permuted=old.reshape(q0+1,q1+1,q2+1,-1).transpose(2,0,1,3).reshape(old.shape)
        assert source==old_channels and target==new_channels
        np.testing.assert_allclose(new.T @ permuted,block,atol=1.6e-15,rtol=0)
        np.testing.assert_allclose(block.T @ block,np.eye(len(source)),atol=2e-15,rtol=0)


@lru_cache(maxsize=1)
def _physical_nine_spin_basis():
    local,labels=triangle_chiral_schur()
    local_positions={label:i for i,label in enumerate(labels)}
    children=(C3Multiplicity(1,e_pairs=1),C3Multiplicity(3,a1=1))
    triple=np.kron(np.kron(local,local),local)
    columns=[];out_labels=[]
    for total_q in (1,3,5,7,9):
        representation=ternary_c3_representation(children,total_q)
        for m in magnetic_numbers(total_q):
            for reduced_index,(q0,q1,q2,q12,a0,a1,a2) in enumerate(representation.channels):
                column=np.zeros(512,dtype=complex)
                for m0,m1,m2 in product(magnetic_numbers(q0),magnetic_numbers(q1),magnetic_numbers(q2)):
                    if m0+m1+m2!=m or abs(m0+m1)>q12:
                        continue
                    coefficient=(clebsch_gordan(q0,m0,q1,m1,q12,m0+m1)
                        *clebsch_gordan(q12,m0+m1,q2,m2,total_q,m))
                    if coefficient:
                        index=[]
                        for q,magnetic,a in ((q0,m0,a0),(q1,m1,a1),(q2,m2,a2)):
                            ell=(1,-1)[a] if q==1 else 0
                            index.append(local_positions[q,magnetic,ell])
                        column[index[0]*64+index[1]*8+index[2]]+=coefficient
                columns.append(column);out_labels.append((total_q,m,reduced_index))
    return triple @ np.column_stack(columns),tuple(out_labels)


def test_all_ordered_charge_blocks_form_exact_physical_SU2_C3_D3_T_representation():
    u,labels=_physical_nine_spin_basis()
    np.testing.assert_allclose(u.conj().T @ u,np.eye(512),atol=6e-15,rtol=0)
    cycle=_physical_image(9,[3*((p+1)%3)+(i+1)%3 for p in range(3) for i in range(3)])
    swap_port=[1,0,2];swap_site=[1,0,2]
    mirror=_physical_image(9,[3*swap_port[p]+swap_site[i] for p in range(3) for i in range(3)])
    children=(C3Multiplicity(1,e_pairs=1),C3Multiplicity(3,a1=1))
    for q in (1,3,5,7,9):
        representation=ternary_c3_representation(children,q)
        high=u[:,[i for i,(total,m,_) in enumerate(labels) if total==q and m==q]]
        low=u[:,[i for i,(total,m,_) in enumerate(labels) if total==q and m==-q]]
        np.testing.assert_allclose(high.conj().T @ _permuted(high,cycle),representation.rotation,atol=2.5e-15,rtol=0)
        np.testing.assert_allclose(high.conj().T @ _permuted(high,mirror),representation.reflection,atol=2.5e-15,rtol=0)
        phase=np.asarray([(-1.)**p.bit_count() for p in range(512)])
        reversed_high=phase[:,None]*high[np.arange(512)^511].conj()
        np.testing.assert_allclose(low.conj().T @ reversed_high,representation.time_reversal_multiplicity,atol=2.5e-15,rtol=0)
        np.testing.assert_allclose(representation.rotation @ representation.rotation @ representation.rotation,
                                  np.eye(len(high.T)),atol=3e-15,rtol=0)


def test_learned_synthesis_is_equivariant_and_has_no_redundant_raw_parameters():
    children=(C3Multiplicity(1,e_pairs=1),C3Multiplicity(3,a1=1))
    source=ternary_c3_representation(children,1);target=C3Multiplicity(1,e_pairs=1)
    rng=np.random.default_rng(20261015)
    raw=rng.normal(size=(len(source.channels),target.width))+1j*rng.normal(size=(len(source.channels),target.width))
    c3=project_ternary_synthesis(raw,source,target)
    np.testing.assert_allclose(source.rotation @ c3,c3 @ target.rotation(),atol=3e-15,rtol=0)
    np.testing.assert_allclose(source.time_reversal_multiplicity @ c3.conj(),
                              c3 @ target.time_reversal_multiplicity(),atol=3e-15,rtol=0)
    assert np.linalg.norm(source.reflection @ c3-c3 @ target.reflection())>1
    w=project_ternary_synthesis(raw,source,target,reflection=True)
    np.testing.assert_allclose(source.rotation @ w,w @ target.rotation(),atol=3e-15,rtol=0)
    np.testing.assert_allclose(source.reflection @ w,w @ target.reflection(),atol=3e-15,rtol=0)
    np.testing.assert_allclose(source.time_reversal_multiplicity @ w.conj(),w @ target.time_reversal_multiplicity(),atol=3e-15,rtol=0)
    basis=ternary_intertwiner_basis(source,target,reflection=True)
    assert len(basis)==15
    # Reflection is optional and additional; do not count its reduction as
    # a consequence of C3 alone. Physical T retains opposite chiralities.
    assert len(ternary_intertwiner_basis(source,target))==30
    assert len(ternary_intertwiner_basis(source,target,reflection=False,time_reversal=False))==60
    real_columns=np.column_stack([np.r_[b.real.ravel(),b.imag.ravel()] for b in basis])
    np.testing.assert_allclose(real_columns.T @ real_columns,np.eye(15),atol=2e-15,rtol=0)
    raw_coordinates=2*len(source.channels)*target.width
    assert raw_coordinates==168 and raw_coordinates>len(basis)
    u,labels=_physical_nine_spin_basis()
    high=u[:,[i for i,(q,m,_) in enumerate(labels) if q==1 and m==1]]
    cycle=_physical_image(9,[3*((p+1)%3)+(i+1)%3 for p in range(3) for i in range(3)])
    decoded=high @ w
    np.testing.assert_allclose(_permuted(decoded,cycle),decoded @ target.rotation(),atol=2e-15,rtol=0)


def _six_spin_singlets():
    u,labels=triangle_chiral_schur();positions={label:i for i,label in enumerate(labels)}
    columns=[]
    for q in (1,3):
        for ell0,ell1 in product((1,-1) if q==1 else (0,),repeat=2):
            column=np.zeros(64,complex)
            for m in magnetic_numbers(q):
                column+=clebsch_gordan(q,m,q,-m,0,0)*np.kron(u[:,positions[q,m,ell0]],u[:,positions[q,-m,ell1]])
            columns.append(column)
    return np.column_stack(columns)


def test_common_C3_and_independent_triangle_invariance_are_not_the_same_physics():
    basis=_six_spin_singlets()
    np.testing.assert_allclose(basis.conj().T @ basis,np.eye(5),atol=1e-15,rtol=0)
    cycle0=_physical_image(6,[1,2,0,3,4,5]);cycle1=_physical_image(6,[0,1,2,4,5,3])
    joint=_physical_image(6,[1,2,0,4,5,3]);mirror=_physical_image(6,[1,0,2,4,3,5])
    joint_projected=(basis+_permuted(basis,joint)+_permuted(_permuted(basis,joint),joint))/3
    full_d3=(joint_projected+_permuted(joint_projected,mirror))/2
    independent=(basis+_permuted(basis,cycle0)+_permuted(_permuted(basis,cycle0),cycle0))/3
    independent=(independent+_permuted(independent,cycle1)+_permuted(_permuted(independent,cycle1),cycle1))/3
    assert np.linalg.matrix_rank(joint_projected,tol=1e-12)==3
    assert np.linalg.matrix_rank(full_d3,tol=1e-12)==2
    assert np.linalg.matrix_rank(independent,tol=1e-12)==1
    # Independent triangle A1 deletes both low-energy q=1 doublets.
    assert np.linalg.norm(independent[:,:4])<2e-15
    h=np.zeros((64,64),complex)
    for first,second in [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3)]:
        image=list(range(6));image[first],image[second]=image[second],image[first]
        p=np.eye(64)[np.argsort(_physical_image(6,image))]
        h+=.5*p-.25*np.eye(64)
    local_energies=np.diag(basis.conj().T @ h @ basis).real
    np.testing.assert_allclose(local_energies,[-1.5,-1.5,-1.5,-1.5,1.5],atol=2e-15,rtol=0)


def test_ternary_joint_map_avoids_the_binary_intermediate_width_two_rank_limit():
    source=ternary_c3_representation((C3Multiplicity(1,e_pairs=1),),1)
    target=C3Multiplicity(1,e_pairs=1)
    rng=np.random.default_rng(20261015)
    w=project_ternary_synthesis(rng.normal(size=(16,2))+1j*rng.normal(size=(16,2)),source,target,reflection=True)
    for selected in (0,2):
        block=np.zeros((2,2,2,2),complex)
        for index,(_,_,_,q12,a0,a1,a2) in enumerate(source.channels):
            if q12==selected:
                block[a0,a1,a2]=w[index]
        assert np.linalg.matrix_rank(block.reshape(4,4),tol=1e-12)==3
        left,singular,right=np.linalg.svd(block.reshape(4,4),full_matrices=False)
        # Complete intermediate features recover the SAME tensor exactly.
        np.testing.assert_allclose((left*singular) @ right,block.reshape(4,4),atol=1e-15,rtol=0)
    assert len(ternary_intertwiner_basis(source,target,reflection=True))==6


def test_width_sixteen_complete_binary_six_site_merge_already_spans_the_ternary_family():
    # Three physical triangles have no intrinsic ternary capacity advantage
    # when the binary six-site intermediate retains every multiplicity.
    widths={1:2,3:1}
    pair_channels={q:[] for q in (0,2,4,6)}
    for q0,q1 in product(widths,repeat=2):
        for q12 in range(abs(q0-q1),q0+q1+1,2):
            pair_channels[q12].extend((q0,q1,a0,a1)
                for a0,a1 in product(range(widths[q0]),range(widths[q1])))
    assert [len(pair_channels[q]) for q in (0,2,4,6)]==[5,9,5,1]
    source=ternary_c3_representation((C3Multiplicity(1,e_pairs=1),C3Multiplicity(3,a1=1)),1)
    rng=np.random.default_rng(20261016)
    w=rng.normal(size=(len(source.channels),16))+1j*rng.normal(size=(len(source.channels),16))
    second={}
    for index,(q0,q1,q2,q12,a0,a1,a2) in enumerate(source.channels):
        intermediate=pair_channels[q12].index((q0,q1,a0,a1))
        second[q12,q2,intermediate,a2]=w[index].copy()
    reconstructed=np.asarray([second[q12,q2,pair_channels[q12].index((q0,q1,a0,a1)),a2]
        for q0,q1,q2,q12,a0,a1,a2 in source.channels])
    np.testing.assert_array_equal(reconstructed,w)


def test_physical_amplitude_and_local_H_need_no_global_fusion_Hamiltonian():
    # A triangular prism has a common D3 action, not independent triangle
    # permutations. Its small exact spectrum is only an amplitude/H oracle.
    basis=_six_spin_singlets()
    joint=_physical_image(6,[1,2,0,4,5,3])
    mirror=_physical_image(6,[1,0,2,4,3,5])
    projected=(basis+_permuted(basis,joint)
               +_permuted(_permuted(basis,joint),joint))/3
    projected=(projected+_permuted(projected,mirror))/2
    u,s,_=np.linalg.svd(projected,full_matrices=False)
    a1=u[:,s>1e-12]
    bonds=((0,1,1.),(1,2,1.),(2,0,1.),(3,4,1.),(4,5,1.),(5,3,1.),
           (0,3,.2),(1,4,.2),(2,5,.2))
    h=np.zeros((64,64),complex)
    for first,second,coupling in bonds:
        image=list(range(6));image[first],image[second]=image[second],image[first]
        p=np.eye(64)[np.argsort(_physical_image(6,image))]
        h+=coupling*(.5*p-.25*np.eye(64))
    # A1 is a symmetry sector of H, unlike independent local A1.
    np.testing.assert_allclose(h[np.ix_(np.argsort(joint),np.argsort(joint))],h,atol=1e-15,rtol=0)
    independent=_physical_image(6,[1,2,0,3,4,5])
    assert np.linalg.norm(h[np.ix_(np.argsort(independent),np.argsort(independent))]-h)>1
    energies,vectors=np.linalg.eigh(a1.conj().T @ h @ a1)
    np.testing.assert_allclose(energies[0],np.linalg.eigvalsh(h)[0],atol=2e-15,rtol=0)
    psi=a1 @ vectors[:,0]
    coefficients=basis.conj().T @ psi
    total_spins=[sum(site[axis] for site in _spin_half_operators(6)) for axis in range(3)]
    casimir=sum(op @ op for op in total_spins)
    np.testing.assert_allclose(np.vdot(psi,casimir @ psi),0,atol=2e-15,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(basis.conj().T @ psi)**2,1,atol=2e-15,rtol=0)
    np.testing.assert_allclose(np.linalg.norm(a1.conj().T @ psi)**2,1,atol=2e-15,rtol=0)
    assert abs(np.vdot(basis[:,-1],psi))**2<.005
    local,labels=triangle_chiral_schur();positions={label:i for i,label in enumerate(labels)}

    def amplitude(packed):
        # Evaluate CG structure and learned reduced coefficients directly for
        # physical bits. No 64-component wavefunction lookup or alpha-space H.
        left,right=packed>>3,packed&7
        total=0j;column=0
        for q in (1,3):
            for ell0,ell1 in product((1,-1) if q==1 else (0,),repeat=2):
                for m in magnetic_numbers(q):
                    total+=(coefficients[column]*clebsch_gordan(q,m,q,-m,0,0)
                            *local[left,positions[q,m,ell0]]
                            *local[right,positions[q,-m,ell1]])
                column+=1
        return total

    evaluated=np.asarray([amplitude(packed) for packed in range(64)])
    np.testing.assert_allclose(evaluated,psi,atol=1e-15,rtol=0)
    action=[]
    for packed in range(64):
        value=0j
        for first,second,coupling in bonds:
            bit0=(packed>>(5-first))&1;bit1=(packed>>(5-second))&1
            value+=coupling*(.25 if bit0==bit1 else -.25)*amplitude(packed)
            if bit0!=bit1:
                flipped=packed^(1<<(5-first))^(1<<(5-second))
                value+=.5*coupling*amplitude(flipped)
        action.append(value)
    action=np.asarray(action)
    np.testing.assert_allclose(action,h @ psi,atol=1.5e-15,rtol=0)
    nonzero=abs(evaluated)>1e-12
    np.testing.assert_allclose(action[nonzero]/evaluated[nonzero],energies[0],atol=1e-14,rtol=0)
    np.testing.assert_allclose(action[~nonzero],0,atol=1e-15,rtol=0)
    # Forcing every triangle to A1 discards all q=1 doublets.
    np.testing.assert_allclose(np.vdot(basis[:,-1],h @ basis[:,-1]),1.25,atol=2e-15,rtol=0)


def test_constructive_N36_cover_has_four_compact_C3_nodes_with_exact_frames():
    # Independently reproducible certificate, not a JSON-dependent golden
    # assertion. Distinct node centers do NOT imply a global tree symmetry.
    nodes=(
        ((0,1),((0,6,7),(31,1,2),(29,35,30))),
        ((2,2),((21,27,28),(26,32,33),(34,4,5))),
        ((5,5),((3,9,10),(8,14,15),(16,22,23))),
        ((3,4),((11,17,12),(18,24,25),(13,19,20))),
    )
    steps=((1,0),(0,1),(1,1),(-1,0),(0,-1),(-1,-1))
    def translate(site,step):
        row,col=divmod(site,6)
        return ((row+step[0])%6)*6+(col+step[1])%6
    neighbors={s:[(translate(s,step),step) for step in steps] for s in range(36)}
    first_shell={tuple(sorted((s,t))) for s in range(36) for t,_ in neighbors[s]}
    second_shell={tuple(sorted((s,translate(s,step)))) for s in range(36)
                  for step in ((1,-1),(1,2),(2,1))}
    whole=[s for _,patches in nodes for patch in patches for s in patch]
    assert sorted(whole)==list(range(36))
    supports=[]
    for (tr,tc),patches in nodes:
        def rotation(site):
            row,col=divmod(site,6)
            return ((-col+tr)%6)*6+(row-col+tc)%6
        assert all(rotation(rotation(rotation(s)))==s for s in range(36))
        assert {tuple(sorted((rotation(a),rotation(b)))) for a,b in first_shell}==first_shell
        assert {tuple(sorted((rotation(a),rotation(b)))) for a,b in second_shell}==second_shell
        for port,patch in enumerate(patches):
            # All twelve registered patches are CCW D triangles; true even
            # where the site labels wrap across a coordinate seam.
            assert patch[1]==translate(patch[0],(1,0))
            assert patch[2]==translate(patch[0],(1,1))
            for local,site in enumerate(patch):
                assert rotation(site)==patches[(port+1)%3][(local+1)%3]
        support=frozenset(s for patch in patches for s in patch);supports.append(support)
        edges=sum(a in support and b in support for a,b in first_shell)
        assert edges==15 and 6*len(support)-2*edges==24
        # Consistent integer lift proves no noncontractible NN cycle. Each
        # branch is genuinely connected, rather than three distant triangles.
        start=min(support);lift={start:divmod(start,6)};todo=[start]
        for site in todo:
            for target,(dr,dc) in neighbors[site]:
                if target not in support:
                    continue
                proposed=(lift[site][0]+dr,lift[site][1]+dc)
                if target not in lift:
                    lift[target]=proposed;todo.append(target)
                else:
                    assert lift[target]==proposed
        assert set(lift)==support
        diameter=0
        for start in support:
            distance={start:0};todo=[start]
            for site in todo:
                for target,_ in neighbors[site]:
                    if target in support and target not in distance:
                        distance[target]=distance[site]+1;todo.append(target)
            diameter=max(diameter,max(distance.values()))
        assert diameter==3
    # No translated global C3 permutes these four chosen coarse supports;
    # one cannot reduce Reynolds images based on local certificates alone.
    for tr,tc in product(range(6),repeat=2):
        def rotation(site):
            row,col=divmod(site,6)
            return ((-col+tr)%6)*6+(row-col+tc)%6
        assert {frozenset(rotation(s) for s in support) for support in supports}!=set(supports)


def test_invalid_quantum_numbers_and_partial_irrep_allocations_reject():
    for q in (True,1.5,-1):
        with pytest.raises((TypeError,ValueError)):
            C3Multiplicity(q,a1=1)
    for widths in ({'a1':0},{'e_pairs':-1},{'a1':True}):
        with pytest.raises(ValueError):
            C3Multiplicity(1,**widths)
    for m in (True,.5,2):
        with pytest.raises((TypeError,ValueError)):
            ternary_coupled_basis(1,1,1,1,m)
