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
    nroots=10,
    basis_per_root=20,
    collapse_per_root=2,
    max_cycle=200,
    tol_s=1e-10,
    tol_semi=1e-10,
    method_type="cvs_ip",
)


def test_niupy_full():
    eom_dsrg.kernel_full()

    e = eom_dsrg.evals
    p = eom_dsrg.spec_info

    assert e[0] == pytest.approx(545.9415248558, abs=1e-8)
    assert e[1] == pytest.approx(571.7665663112, abs=1e-8)
    assert e[2] == pytest.approx(572.0315519843, abs=1e-8)
    assert e[3] == pytest.approx(572.2423427205, abs=1e-8)
    assert e[8] == pytest.approx(574.5615314963, abs=1e-8)

    assert p[0] == pytest.approx(1.61761416, abs=1e-6)
    assert p[1] == pytest.approx(0.00000000, abs=1e-6)
    assert p[2] == pytest.approx(0.00000000, abs=1e-6)
    assert p[3] == pytest.approx(0.00000000, abs=1e-6)
    assert p[8] == pytest.approx(0.01134412, abs=1e-6)


def test_niupy():
    eom_dsrg.kernel()

    e = eom_dsrg.evals
    p = eom_dsrg.spec_info

    assert e[0] == pytest.approx(545.9415248558, abs=1e-8)
    assert e[1] == pytest.approx(571.7665663109, abs=1e-8)
    assert e[2] == pytest.approx(572.0315519842, abs=1e-8)
    assert e[3] == pytest.approx(572.2423427206, abs=1e-8)
    assert e[8] == pytest.approx(574.5615314963, abs=1e-8)

    assert p[0] == pytest.approx(1.61761416, abs=1e-6)
    assert p[1] == pytest.approx(0.00000000, abs=1e-6)
    assert p[2] == pytest.approx(0.00000000, abs=1e-6)
    assert p[3] == pytest.approx(0.00000000, abs=1e-6)
    assert p[8] == pytest.approx(0.01134412, abs=1e-6)
