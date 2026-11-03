import hashlib
import json
import pytest
from dpa4c_sog import warmstart_provenance as g


def sha(b): return hashlib.sha256(b).hexdigest()


def fixture(tmp_path, monkeypatch):
    run=tmp_path/'run'; run.mkdir()
    prep=tmp_path/'prep'; prep.mkdir()
    data=tmp_path/'data'; data.mkdir()
    for split in ('train','val'):
        (data/(split+'.lmdb')).mkdir()
        (data/(split+'.lmdb')/'data.mdb').write_bytes(split.encode())
    monkeypatch.setattr(g,'TRAIN_SHA',sha(b'train')); monkeypatch.setattr(g,'VAL_SHA',sha(b'val'))
    cfg={'training':{'training_data':{'systems':str(data/'train.lmdb'),'batch_size':8,'auto_prob_style':'prob_sys_size'},
         'validation_data':{'systems':str(data/'val.lmdb'),'batch_size':1},'seed':42,'num_steps':442978,'change_bias_after_training':False}}
    cb=json.dumps(cfg).encode(); (run/'restricted_input.json').write_bytes(cb)
    info={'config_sha256':sha(cb),'config_path':str(prep/'input_dpa4c_lmdb.json'),
          'train_path':str(data/'train.lmdb'),'val_path':str(data/'val.lmdb'),'train_sha256':g.TRAIN_SHA,'val_sha256':g.VAL_SHA,
          'initial_sha256':g.PILOT_SHA,'local_sources':{'restricted_train_batch8_experimental.py':g.B8_CODE_SHA}}
    fp=sha(json.dumps(info,sort_keys=True,separators=(',',':')).encode())
    (run/'restricted_identity.json').write_text(json.dumps({'fingerprint':fp,'identity':info}))
    cp=run/'model.ckpt-245000.pt'; cp.write_bytes(b'model')
    side={'version':1,'step':245000,'fingerprint':fp,'checkpoint_sha256':sha(b'model'),
          'batch_counts':{'train':245000,'val':246},'seeds':{'train':42,'val':42}}
    (run/'model.ckpt-245000.pt.sampler.json').write_text(json.dumps(side))
    (prep/'preparation.json').write_text(json.dumps({'source_index_identity':g.SOURCE_ID,'debug_only':False,
       'no_training_executed':True,'test_split_read':False,'source_counts':{'train':3543786,'val':443734},'config_sha256':sha(cb)}))
    (data/'status.json').write_text(json.dumps({'status':'complete','no_sampling':True,'test_split_read':False,
       'source_index_metadata_sha256':g.SOURCE_ID,'counts':{'train':3543786,'val':443734},
       'train_lmdb_sha256':g.TRAIN_SHA,'val_lmdb_sha256':g.VAL_SHA}))
    return run,prep,data,sha(b'model'),fp


def test_verified_and_tampering_fails(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    assert g.verify(*args)['test_split_read'] is False
    with pytest.raises(ValueError,match='SHA'): g.verify(*args[:3],'0'*64,args[4])
    (args[0]/'model.ckpt-245000.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'): g.verify(*args)


def test_rejects_changed_config_and_data(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    (args[0]/'restricted_input.json').write_text('{}')
    with pytest.raises(ValueError,match='config'): g.verify(*args)


def test_rejects_sidecar_and_test_provenance(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    p=args[0]/'model.ckpt-245000.pt.sampler.json'; obj=json.loads(p.read_text()); obj['batch_counts']['train']=1; p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='progress'): g.verify(*args)


def test_rejects_test_split_provenance(tmp_path,monkeypatch):
    args=fixture(tmp_path,monkeypatch)
    p=args[2]/'status.json'; obj=json.loads(p.read_text()); obj['test_split_read']=True; p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='status'): g.verify(*args)
