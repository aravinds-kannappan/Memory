import zipfile
import numpy as np
import pandas as pd
from consent_ai.data import upfall_skeleton,ntu_skeleton

def test_upfall_actual_filename_and_header_variants(tmp_path):
    cols=[f'Joint{j}_{axis}' for j in range(1,34) for axis in 'XYZ']
    with zipfile.ZipFile(tmp_path/'SUBJECT1.zip','w') as z:
        for name,label in [('C2S1_A1_T2.csv','LLABEL'),('C1S1A1T3.csv','0')]:
            d=pd.DataFrame(np.zeros((4,99)),columns=cols);d[label]=[0,1,1,0]
            z.writestr(name,d.to_csv(index=False))
    r,a=upfall_skeleton(tmp_path)
    assert len(r)==2 and {x['trial'] for x in r}=={2,3}
    assert {x['camera'] for x in a}=={1,2}

def test_ntu_published_frame_body_joint_layout(tmp_path):
    p=tmp_path/'S001C002P003R001A043.skeleton'
    p.write_text('1\n1\n42 0 0 0 0 0 0 0 0 0\n25\n'+'1 2 3 0 0 0 0 0 0 0 0 0\n'*25)
    r=ntu_skeleton(p)
    assert r['x'].shape==(1,25,3) and r['subject']==3 and r['activity']==43
