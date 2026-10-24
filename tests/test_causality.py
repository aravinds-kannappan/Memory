import pandas as pd
from consent_ai.ehr import summarize, split_units

def test_future_event_and_late_record_cannot_change_features():
    cutoff=pd.Timestamp('2020-01-01 12:00');start=pd.Timestamp('2020-01-01')
    past=pd.DataFrame({'itemid':[1],'charttime':[pd.Timestamp('2020-01-01 10:00')],'storetime':[pd.Timestamp('2020-01-01 11:00')],'valuenum':[5.]})
    future=pd.DataFrame({'itemid':[1,1],'charttime':[pd.Timestamp('2020-01-01 13:00'),pd.Timestamp('2020-01-01 09:00')],'storetime':[pd.Timestamp('2020-01-01 13:00')]*2,'valuenum':[999.,999.]})
    assert summarize(past,cutoff,start,[1],'v')==summarize(pd.concat([past,future]),cutoff,start,[1],'v')

def test_group_splits_disjoint_for_repeated_rows():
    d=pd.DataFrame({'patient':list(range(100))*3,'era':['early']*50+['late']*50+['early']*50+['late']*50+['early']*50+['late']*50})
    for mode in ['random','era_shift']:
        s=split_units(d,11,mode)
        assert not set(s['train'])&set(s['test'])
        assert not set(s['calibration'])&set(s['test'])
