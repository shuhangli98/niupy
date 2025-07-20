import pytest
import niupy

mo_spaces = {
    "FROZEN_DOCC": [1, 0, 0, 0],
    "RESTRICTED_DOCC": [1, 0, 0, 1],
    "ACTIVE": [2, 0, 1, 1],
}

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=True,
    mo_spaces=mo_spaces,
    nroots=4,
    basis_per_root=50,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-5,
    method_type="cvs_ee",
)


def test_niupy_full():
    eom_dsrg.kernel_full()

    e = eom_dsrg.evals
    p = eom_dsrg.spec_info

    assert e[0] == pytest.approx(540.2887714820, abs=1e-8)
    assert e[1] == pytest.approx(540.9660608351, abs=1e-8)
    assert e[2] == pytest.approx(542.4188308903, abs=1e-8)
    assert e[3] == pytest.approx(543.3395502334, abs=1e-8)

    assert p[0] == pytest.approx(0.00000000, abs=1e-8)
    assert p[1] == pytest.approx(0.01996767, abs=1e-8)
    assert p[2] == pytest.approx(0.00000000, abs=1e-8)
    assert p[3] == pytest.approx(0.04296089, abs=1e-8)


def test_niupy():
    eom_dsrg.kernel()

    e = eom_dsrg.evals
    p = eom_dsrg.spec_info

    assert e[0] == pytest.approx(540.2887714846, abs=1e-8)
    assert e[1] == pytest.approx(540.9660608382, abs=1e-8)
    assert e[2] == pytest.approx(542.4188308949, abs=1e-8)
    assert e[3] == pytest.approx(543.3395502382, abs=1e-8)

    assert p[0] == pytest.approx(0.00000000, abs=1e-6)
    assert p[1] == pytest.approx(0.01991211, abs=1e-6)
    assert p[2] == pytest.approx(0.00000000, abs=1e-6)
    assert p[3] == pytest.approx(0.04335803, abs=1e-6)
