import hashlib
import json

import pytest
from cace_sog_training.data import IndexReader
from dpa4c_sog.validate import validate
from dpa4c_sog.validate_shards import shard, merge
from dpa4c_sog.predict import _verified_validation, _validation_frames
from test_dpa4c_validate import inputs, ScaleModel


def build_parts(tmp_path, db, files):
    parts = []
    models = {files[0]: ScaleModel('dpa', 1), files[1]: ScaleModel('dpa', 0)}
    for index, ckpt in enumerate(files):
        for j, (a, b) in enumerate(((0, 1), (1, 3))):
            out = tmp_path / f'{index}-{j}.jsonl'
            shard(db, ckpt, a, b, out, loader=models.get)
            parts.append(out)
    return parts, models


def test_matches_monolithic_and_validation_gate(tmp_path, monkeypatch):
    db, files = inputs(tmp_path)
    orig = IndexReader.record
    def read_only_val(self, channel, split, ordinal):
        assert channel == 'sog_candidates' and split == 'val'
        return orig(self, channel, split, ordinal)
    monkeypatch.setattr(IndexReader, 'record', read_only_val)
    parts, models = build_parts(tmp_path, db, files)
    result = merge(db, files, parts[::-1], tmp_path / 'merged')
    full = validate(db, 'dpa', files, tmp_path / 'full', (0, 1), loader=models.get)
    assert result['selected_index'] == full['selected_index'] == 1
    assert result['frame_identity_sha256'] == full['frame_identity_sha256']
    for a, b in zip(result['candidates'], full['candidates']):
        for prop in ('energy', 'force'):
            for key in ('count', 'mean_difference', 'mae', 'rmse', 'r2'):
                assert a[prop][key] == pytest.approx(b[prop][key])
    with IndexReader(db) as reader:
        n, digest = _validation_frames(reader)
    _verified_validation(tmp_path / 'merged/validation.json', 'dpa', result['selected_checkpoint_sha256'],
                         result['index_identity'], n, digest)
    assert not result['test_split_read']


def test_missing_overlap_wrong_checkpoint_and_tamper_fail(tmp_path):
    db, files = inputs(tmp_path)
    parts, _ = build_parts(tmp_path, db, files)
    for name, selected, message in [('missing', parts[:-1], 'full indexed'),
                                     ('overlap', parts + [parts[0]], 'overlap')]:
        with pytest.raises(ValueError, match=message):
            merge(db, files, selected, tmp_path / name)
        assert not (tmp_path / name / 'validation.json').exists()
    wrong = tmp_path / 'wrong.pt'; wrong.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='unrequested checkpoint'):
        merge(db, [files[0], wrong], parts, tmp_path / 'wrong')
    parts[1].write_bytes(parts[1].read_bytes() + b'{}\n')
    with pytest.raises(ValueError):
        merge(db, files, parts, tmp_path / 'tampered')
    assert not (tmp_path / 'tampered/validation.json').exists()


def test_bad_ranges_and_no_overwrite(tmp_path):
    db, files = inputs(tmp_path)
    with pytest.raises(ValueError, match='ordinal range'):
        shard(db, files[0], 0, 4, tmp_path / 'bad', loader=lambda _: ScaleModel('dpa', 0))
    assert json.loads((tmp_path / 'bad.status.json').read_text())['status'] == 'failed'
    shard(db, files[0], 0, 1, tmp_path / 'part', loader=lambda _: ScaleModel('dpa', 0))
    with pytest.raises(FileExistsError):
        shard(db, files[0], 0, 1, tmp_path / 'part', loader=lambda _: ScaleModel('dpa', 0))
