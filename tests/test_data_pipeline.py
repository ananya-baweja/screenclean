import numpy as np
import pytest
from conftest import make_uhdm_tree

from screenclean.data import shards, transforms, uhdm
from screenclean.utils.drive import sha1_file, stage_shards

torch = pytest.importorskip("torch")
from screenclean.data.pairs_dataset import PairsDataset  # noqa: E402

# --------------------------------------------------------------------------- uhdm


def test_discover_pairs_keys_and_problems(tmp_path):
    keys = make_uhdm_tree(tmp_path / "train", {"pair_00": 2, "pair_01": 2})
    (tmp_path / "train" / "pair_01" / "0009_gt.jpg").write_bytes(b"x")  # no moire partner
    names = uhdm.list_files(tmp_path)
    pairs, problems = uhdm.discover_pairs(names)
    assert [p.key for p in pairs] == ["train/" + k for k in keys]
    assert pairs[0].gt == "train/pair_00/0000_gt.jpg" and pairs[0].moire == "train/pair_00/0000_moire.jpg"
    assert problems == ["train/pair_01/0009: ground truth without a moire image"]
    # mirror-style names: the prefix is stripped from keys, other folders are ignored
    mirror = [
        "train/train/pair_22/0254_gt.jpg",
        "train/train/pair_22/0254_moire.jpg",
        "test/test/0001_gt.jpg",
    ]
    pairs, problems = uhdm.discover_pairs(mirror, "train/")
    assert [p.key for p in pairs] == ["train/pair_22/0254"] and problems == []


def test_splits_are_deterministic_and_disjoint():
    keys = [f"k{i:03d}" for i in range(50)]
    shuffled = list(reversed(keys))
    train, val = uhdm.split_train_val(shuffled, 10, seed=0)
    assert (train, val) == uhdm.split_train_val(keys, 10, seed=0)  # input order doesn't matter
    assert len(val) == 10 and not set(train) & set(val) and sorted(train + val) == keys
    assert uhdm.sample_keys(keys, 10, seed=1) != val
    with pytest.raises(ValueError):
        uhdm.sample_keys(keys, 51)


def test_crops_are_aligned_and_deterministic():
    rng = np.random.default_rng(0)
    gt = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    moire = gt.copy()
    crops = uhdm.train_crops(moire, gt, "a/0001", size=64, n_full=2, n_half=1)
    assert len(crops) == 3
    for m, g in crops:
        assert m.shape == (64, 64, 3)
        np.testing.assert_array_equal(m, g)  # same window in both images
    again = uhdm.train_crops(moire, gt, "a/0001", size=64, n_full=2, n_half=1)
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(crops, again, strict=True))
    other = uhdm.train_crops(moire, gt, "a/0002", size=64, n_full=2, n_half=1)
    assert not np.array_equal(crops[0][0], other[0][0])
    assert len(uhdm.val_crops(moire, gt, "a/0001", size=64, n=2)) == 2


def test_crop_rejects_mismatched_or_small_pairs():
    rng = np.random.default_rng(0)
    a = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError, match="differ"):
        transforms.random_crop_pair(a, np.zeros((100, 90, 3), np.uint8), 32, rng)
    with pytest.raises(ValueError, match="smaller"):
        transforms.random_crop_pair(a, a, 128, rng)


def test_flip_rot_pair_keeps_pairs_together():
    rng = np.random.default_rng(3)
    a = np.arange(5 * 7 * 3).reshape(5, 7, 3)
    for _ in range(20):
        x, y = transforms.flip_rot_pair(a, a + 1, rng)
        np.testing.assert_array_equal(x + 1, y)


# --------------------------------------------------------------------------- shards


def _samples(n, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        yield shards.sample_id(f"train/pair_0{i}/0001", 0), rng.bytes(100 + i), rng.bytes(50)


def test_sample_id_roundtrip():
    sid = shards.sample_id("train/pair_22/0254", 3)
    assert sid == "train__pair_22__0254--c3"
    assert shards.source_key(sid) == "train/pair_22/0254"
    assert shards.source_key(shards.sample_id("test/0007")) == "test/0007"


def test_write_index_read_shard(tmp_path):
    items = list(_samples(3))
    entry = shards.write_shard(tmp_path / "s-00000.tar", items)
    assert entry["samples"] == 3 and entry["sha1"] == sha1_file(tmp_path / "s-00000.tar")
    index = shards.index_tar(tmp_path / "s-00000.tar")
    groups = shards.group_samples(index)
    assert sorted(groups) == sorted(s for s, _, _ in items)
    sid, moire, gt = items[1]
    assert shards.read_member(tmp_path / "s-00000.tar", *index[groups[sid]["moire"]]) == moire
    assert shards.read_member(tmp_path / "s-00000.tar", *index[groups[sid]["gt"]]) == gt
    # identical input -> identical bytes (so re-runs are reproducible)
    assert shards.write_shard(tmp_path / "copy.tar", items)["sha1"] == entry["sha1"]


def test_manifest_and_stage_shards(tmp_path):
    drive = tmp_path / "drive" / "train"
    m = shards.load_manifest(drive)
    for i in range(2):
        entry = shards.write_shard(drive / f"train-{i:05d}.tar", _samples(2, seed=i))
        shards.add_shard(drive, m, entry)
    m["complete"] = True
    shards.save_manifest(drive, m)
    assert shards.finished_shards(drive, shards.load_manifest(drive)) == {
        "train-00000.tar",
        "train-00001.tar",
    }

    local = stage_shards(drive, tmp_path / "local", verify_sha1=True)
    assert [p.name for p in local] == ["train-00000.tar", "train-00001.tar"]
    mtime = local[0].stat().st_mtime_ns
    stage_shards(drive, tmp_path / "local")  # second call skips existing copies
    assert local[0].stat().st_mtime_ns == mtime

    (drive / "train-00001.tar").write_bytes(b"truncated")
    assert shards.finished_shards(drive, shards.load_manifest(drive)) == {"train-00000.tar"}


# --------------------------------------------------------------------------- dataset


@pytest.fixture
def pair_shard(tmp_path):
    from screenclean.utils.io import encode_jpeg

    rng = np.random.default_rng(0)
    items = []
    for i in range(4):
        gt = rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)
        items.append((shards.sample_id(f"train/p/{i:04d}", 0), encode_jpeg(gt), encode_jpeg(gt)))
    shards.write_shard(tmp_path / "train-00000.tar", items)
    return tmp_path / "train-00000.tar"


def test_pairs_dataset_shapes_and_alignment(pair_shard):
    ds = PairsDataset([pair_shard], crop=64, augment=True)
    assert len(ds) == 4
    moire, gt = ds[0]
    assert moire.shape == gt.shape == (3, 64, 64) and moire.dtype == torch.float32
    assert 0.0 <= float(moire.min()) and float(moire.max()) <= 1.0
    torch.testing.assert_close(moire, gt)  # identical images stay identical after paired augmentation

    full = PairsDataset([pair_shard])
    assert full[0][0].shape == (3, 96, 128)


def test_pairs_dataset_exclude_and_loader(pair_shard):
    ds = PairsDataset([pair_shard], crop=32, augment=True, exclude_keys={"train/p/0001"})
    assert len(ds) == 3
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=3)))
    assert batch[0].shape == (3, 3, 32, 32)
