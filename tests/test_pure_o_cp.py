import pytest

pytest.importorskip("ortools.sat.python.cp_model")

from python.pure_o_cp import solve_h_vector


def test_pure_o_feasible_case():
    result = solve_h_vector([1, 3, 3], timeout_sec=5.0, num_workers=1)
    assert result.status == "FEASIBLE"


def test_pure_o_infeasible_case():
    # Degree-2 with three degree-1 monomials and only one degree-2 monomial is impossible.
    result = solve_h_vector([1, 3, 1], timeout_sec=5.0, num_workers=1)
    assert result.status in {"INFEASIBLE", "UNKNOWN"}
