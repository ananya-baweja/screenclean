import numpy as np
import pytest

from screenclean.eval.stats import bootstrap_mean, paired_bootstrap, paired_by_key


def test_bootstrap_mean_covers_the_true_mean_and_is_repeatable():
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 1.0, 400)
    ci = bootstrap_mean(x)
    assert ci.lo < 2.0 < ci.hi
    assert ci.mean == pytest.approx(x.mean())
    # standard error of the mean is 0.05, so the 95% interval is about +-0.1
    assert 0.15 < ci.hi - ci.lo < 0.25
    assert bootstrap_mean(x) == ci  # seeded


def test_pairing_removes_item_difficulty():
    rng = np.random.default_rng(2)
    difficulty = rng.normal(0, 5, 200)  # large spread between items
    a = difficulty + 0.3 + rng.normal(0, 0.1, 200)
    b = difficulty
    paired = paired_bootstrap(a, b)
    assert paired.excludes_zero() and paired.lo > 0.25 and paired.hi < 0.35
    # comparing the two means without pairing could not tell them apart
    unpaired = bootstrap_mean(a).lo < bootstrap_mean(b).hi
    assert unpaired


def test_identical_methods_give_a_zero_interval():
    x = [0.1, 0.5, 0.2]
    ci = paired_bootstrap(x, x)
    assert (ci.mean, ci.lo, ci.hi) == (0.0, 0.0, 0.0) and not ci.excludes_zero()


def test_paired_by_key_uses_common_items():
    a = {"p1": 0.2, "p2": 0.4, "p3": 0.9}
    b = {"p1": 0.1, "p2": 0.2}
    ci = paired_by_key(a, b, n_resamples=200)
    assert ci.n == 2 and ci.mean == pytest.approx(0.15)
    with pytest.raises(ValueError):
        paired_by_key({"x": 1.0}, {"y": 1.0})
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1, 2, 3])
