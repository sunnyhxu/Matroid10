from python.run_pipeline import apply_override, parse_override_value


def test_parse_override_value_primitives():
    assert parse_override_value("true") is True
    assert parse_override_value("42") == 42
    assert parse_override_value("3.5") == 3.5
    assert parse_override_value('"abc"') == "abc"


def test_parse_override_value_list():
    assert parse_override_value("[2,3,4]") == [2, 3, 4]


def test_apply_override_nested():
    cfg = {"generation": {"threads": 4}}
    apply_override(cfg, "generation.threads=16")
    apply_override(cfg, "pipeline.max_wall_seconds=600")
    assert cfg["generation"]["threads"] == 16
    assert cfg["pipeline"]["max_wall_seconds"] == 600


def test_apply_override_new_generation_keys():
    cfg = {"generation": {}}
    apply_override(cfg, "generation.mode=sparse_paving")
    apply_override(cfg, "generation.shard_index=1")
    apply_override(cfg, "generation.shard_count=4")
    apply_override(cfg, "generation.trial_index_start=1000")
    apply_override(cfg, "generation.sparse_accept_prob=0.15")
    apply_override(cfg, "generation.sparse_min_circuit_hyperplanes=2")
    apply_override(cfg, "generation.sparse_max_circuit_hyperplanes=8")
    assert cfg["generation"]["mode"] == "sparse_paving"
    assert cfg["generation"]["shard_index"] == 1
    assert cfg["generation"]["shard_count"] == 4
    assert cfg["generation"]["trial_index_start"] == 1000
    assert cfg["generation"]["sparse_accept_prob"] == 0.15
    assert cfg["generation"]["sparse_min_circuit_hyperplanes"] == 2
    assert cfg["generation"]["sparse_max_circuit_hyperplanes"] == 8
