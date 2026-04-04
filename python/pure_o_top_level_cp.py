from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from .pure_o_cp import CPResult, enumerate_monomials, extract_solver_stats, infer_n_vars_from_h_vector
except ImportError:
    from pure_o_cp import CPResult, enumerate_monomials, extract_solver_stats, infer_n_vars_from_h_vector


def _dominates(alpha: Tuple[int, ...], beta: Tuple[int, ...]) -> bool:
    return all(a >= b for a, b in zip(alpha, beta))


def solve_h_vector_top_level(
    h_vector: List[int],
    timeout_sec: float,
    num_workers: int,
    emit_witness: bool = False,
) -> CPResult:
    try:
        from ortools.sat.python import cp_model
    except Exception as ex:  # noqa: BLE001
        return CPResult(
            status="ERROR",
            wall_time=0.0,
            model_size={"num_vars": 0, "num_constraints": 0, "num_degree_buckets": 0},
            error=f"ortools_import_failed:{ex}",
            witness=None,
        )

    if not h_vector or any(x < 0 for x in h_vector):
        return CPResult(
            status="ERROR",
            wall_time=0.0,
            model_size={"num_vars": 0, "num_constraints": 0, "num_degree_buckets": 0},
            error="invalid_h_vector",
            witness=None,
        )

    d = len(h_vector) - 1
    n_vars = infer_n_vars_from_h_vector(h_vector)
    by_deg = enumerate_monomials(n_vars, d)

    for k, hk in enumerate(h_vector):
        if hk > len(by_deg.get(k, [])):
            return CPResult(
                status="INFEASIBLE",
                wall_time=0.0,
                model_size={
                    "num_vars": sum(len(v) for v in by_deg.values()),
                    "num_constraints": 0,
                    "num_degree_buckets": len(by_deg),
                    "num_top_vars": len(by_deg.get(d, [])),
                    "num_coverage_vars": sum(len(by_deg.get(k, [])) for k in range(d)),
                },
                error=f"degree_{k}_count_exceeds_capacity",
                witness=None,
            )

    model = cp_model.CpModel()
    top_monomials = by_deg[d]
    y = {alpha: model.NewBoolVar("top_" + "_".join(str(v) for v in alpha)) for alpha in top_monomials}
    z = {}

    constraints = 0
    model.Add(sum(y.values()) == h_vector[d])
    constraints += 1

    for deg in range(d):
        for beta in by_deg[deg]:
            z[beta] = model.NewBoolVar("div_" + "_".join(str(v) for v in beta))

    for deg in range(d):
        for beta in by_deg[deg]:
            covering_tops = [y[alpha] for alpha in top_monomials if _dominates(alpha, beta)]
            if not covering_tops:
                model.Add(z[beta] == 0)
                constraints += 1
                continue

            model.Add(sum(covering_tops) >= z[beta])
            constraints += 1
            for top_var in covering_tops:
                model.Add(top_var <= z[beta])
                constraints += 1

    for deg in range(d):
        model.Add(sum(z[beta] for beta in by_deg[deg]) == h_vector[deg])
        constraints += 1

    if y:
        model.AddDecisionStrategy(
            list(y.values()),
            cp_model.CHOOSE_FIRST,
            cp_model.SELECT_MAX_VALUE,
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_sec
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    wall_time = float(solver.WallTime())
    solver_stats = extract_solver_stats(solver, wall_time)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        cp_status = "FEASIBLE"
    elif status == cp_model.INFEASIBLE:
        cp_status = "INFEASIBLE"
    elif status == cp_model.UNKNOWN:
        cp_status = "UNKNOWN"
    else:
        cp_status = "ERROR"

    witness = None
    if cp_status == "FEASIBLE" and emit_witness:
        selected_top_monomials = [list(alpha) for alpha, var in y.items() if solver.BooleanValue(var)]
        witness = {
            "n_vars": n_vars,
            "max_degree": d,
            "selected_top_monomials": selected_top_monomials,
        }

    return CPResult(
        status=cp_status,
        wall_time=wall_time,
        model_size={
            "num_vars": len(model.Proto().variables),
            "num_constraints": constraints,
            "num_degree_buckets": len(by_deg),
            "num_top_vars": len(y),
            "num_coverage_vars": len(z),
        },
        witness=witness,
        solver_stats=solver_stats,
    )
