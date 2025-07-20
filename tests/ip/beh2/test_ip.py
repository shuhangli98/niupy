import pytest
import niupy

eom_dsrg = niupy.EOM_DSRG(
    opt_einsum=True,
    nroots=10,
    basis_per_root=20,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="ip",
)


def test_niupy_full():
    eom_dsrg.kernel_full()

    e = eom_dsrg.evals
    p = eom_dsrg.spec_info

    assert e[0] == pytest.approx(11.0712299825, abs=1e-8)
    assert e[1] == pytest.approx(12.9096937411, abs=1e-8)
    assert e[2] == pytest.approx(17.2339756217, abs=1e-8)
    assert e[3] == pytest.approx(17.3971330715, abs=1e-8)

    assert p[0] == pytest.approx(1.97180714, abs=1e-6)
    assert p[1] == pytest.approx(1.95473561, abs=1e-6)
    assert p[2] == pytest.approx(0.00000000, abs=1e-6)
    assert p[3] == pytest.approx(0.00213609, abs=1e-6)


def test_niupy():
    eom_dsrg.kernel()

    e = eom_dsrg.evals
    p = eom_dsrg.spec_info

    assert e[0] == pytest.approx(11.0712299826, abs=1e-8)
    assert e[1] == pytest.approx(12.9096937411, abs=1e-8)
    assert e[2] == pytest.approx(17.2339756217, abs=1e-8)
    assert e[3] == pytest.approx(17.3971330810, abs=1e-8)

    assert p[0] == pytest.approx(1.97180714, abs=1e-6)
    assert p[1] == pytest.approx(1.95473561, abs=1e-6)
    assert p[2] == pytest.approx(0.00000000, abs=1e-6)
    assert p[3] == pytest.approx(0.00213609, abs=1e-6)
