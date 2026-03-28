from python.search_progress import (
    aggregate_progress,
    replace_readme_section,
    render_table,
    upsert_ledger_rows,
)


def test_upsert_ledger_rows_idempotent():
    chunk = {
        "run_id": "r1",
        "phase1_command_index": 0,
        "mode": "representable",
        "category": "representable_f2",
        "field": 2,
        "n": 10,
        "seed": 42,
        "shard_index": 0,
        "shard_count": 2,
        "trial_start": 0,
        "trial_stride": 2,
        "candidates": 10,
    }
    rows = upsert_ledger_rows([], [chunk])
    assert len(rows) == 1
    rows_again = upsert_ledger_rows(rows, [chunk])
    assert len(rows_again) == 1


def test_aggregate_progress_categories_and_yield():
    rows = [
        {"category": "representable_f2", "mode": "representable", "field": 2, "n": 10, "candidates": 1000, "unique_hits": 5},
        {"category": "representable_f3", "mode": "representable", "field": 3, "n": 10, "candidates": 2000, "unique_hits": 0},
        {"category": "sparse_paving", "mode": "sparse_paving", "n": 10, "candidates": 0, "unique_hits": 0},
    ]
    agg = aggregate_progress(rows)
    by_cat = {x["category"]: x for x in agg}
    assert by_cat["representable_f2"]["status"] == "Active"
    assert by_cat["representable_f2"]["trials"] == 1000
    assert by_cat["representable_f2"]["unique"] == 5
    assert by_cat["representable_f2"]["yield_percent"] == 0.5
    assert by_cat["representable_f3"]["status"] == "Sampled"
    assert by_cat["representable_f3"]["unique"] == 0
    assert by_cat["sparse_paving"]["status"] == "Not Started"


def test_replace_readme_section_bounded_markers():
    original = (
        "# Title\n\n"
        "Before section.\n\n"
        "<!-- SEARCH_PROGRESS:START -->\nold\n<!-- SEARCH_PROGRESS:END -->\n\n"
        "After section.\n"
    )
    new_table = render_table(
        [
            {
                "category_label": "Representable $\\mathbb{F}_2$",
                "n": 10,
                "status": "Active",
                "method": "Bitset C++ / CP-SAT",
                "trials_display": "1.0K",
                "unique_display": "5",
                "yield_display": "0.500%",
            }
        ]
    )
    updated = replace_readme_section(original, new_table)
    assert "Before section." in updated
    assert "After section." in updated
    assert "old" not in updated
    assert "Representable $\\mathbb{F}_2$" in updated
