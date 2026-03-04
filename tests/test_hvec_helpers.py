from python.hvec_extract import h_from_f_vector, prefilter_h_vector


def test_h_from_f_vector_simplex():
    # Rank-2 uniform matroid U(2,2): f_-1=1, f_0=2, f_1=1.
    f_vec = [1, 2, 1]
    assert h_from_f_vector(f_vec, rank=2) == [1, 0, 0]


def test_prefilter_h1_le_h2_rejects():
    reasons = prefilter_h_vector([1, 5, 2, 1], check_h1_le_h2=True)
    assert "h1_gt_h2" in reasons


def test_prefilter_accepts_basic():
    reasons = prefilter_h_vector([1, 3, 3, 1], check_h1_le_h2=True)
    assert reasons == []


def test_prefilter_rejects_h_rank_nonpositive():
    reasons = prefilter_h_vector([1, 2, 3, 2, 0], check_h1_le_h2=True, check_h_rank_positive=True)
    assert "h_rank_nonpositive" in reasons
