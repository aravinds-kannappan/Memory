"""Secondary public skeleton experiment: compact ST-GCN and matched ablations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch import nn
from .data import upfall_skeleton
from .evaluation import fit_model,probability,metrics,cluster_bootstrap,paired_bootstrap
from .risk import calibrate

POSE_EDGES=[(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
 (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
 (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),(11,23),(12,24),
 (23,24),(23,25),(25,27),(27,29),(29,31),(27,31),(24,26),(26,28),(28,30),(30,32),(28,32)]

def adjacency(variant,seed=11):
    g=nx.Graph();g.add_nodes_from(range(33));g.add_edges_from(POSE_EDGES)
    if variant=='shuffled':nx.double_edge_swap(g,nswap=50,max_tries=5000,seed=seed)
    A=np.stack([np.eye(33),np.zeros((33,33)),np.zeros((33,33))]).astype(np.float32)
    if variant!='topology_free':
        # Distance partition: separate self and two directed neighbor partitions.
        distance=nx.multi_source_dijkstra_path_length(g,[23,24],weight=None)
        for u,v in g.edges:
            if distance.get(u,99)<=distance.get(v,99):A[1,u,v]=1;A[2,v,u]=1
            else:A[1,v,u]=1;A[2,u,v]=1
    A=A/np.maximum(A.sum(axis=1,keepdims=True),1)
    return A

class Block(nn.Module):
    def __init__(self,cin,cout,A):
        super().__init__();self.register_buffer('A',torch.tensor(A));self.cout=cout
        self.spatial=nn.Conv2d(cin,cout*len(A),1)
        self.temporal=nn.Sequential(nn.BatchNorm2d(cout),nn.ReLU(),nn.Conv2d(cout,cout,(3,1),padding=(1,0)),nn.BatchNorm2d(cout),nn.ReLU())
    def forward(self,x):
        z=self.spatial(x);b,_,t,v=z.shape
        z=z.reshape(b,len(self.A),self.cout,t,v)
        z=torch.einsum('bkctv,kvw->bctw',z,self.A)
        return self.temporal(z)

class STGCN(nn.Module):
    def __init__(self,A,channels=(32,64)):
        super().__init__();self.blocks=nn.Sequential(Block(3,channels[0],A),Block(channels[0],channels[1],A));self.head=nn.Linear(channels[1],1)
    def forward(self,x):return self.head(self.blocks(x).mean(dim=(2,3))).squeeze(-1)

def windows(root,out):
    records,audit=upfall_skeleton(root);pd.DataFrame(audit).to_csv(Path(out,'movement_file_audit.csv'),index=False)
    X=[];rows=[]
    for r in records:
        # 32 observed frames, stride 8. Last-frame impact is contemporaneous,
        # not an assertion of future fall prediction or a known frame duration.
        for end in range(32,len(r['x'])+1,8):
            x=r['x'][end-32:end].copy();root_joint=(x[:,23]+x[:,24])/2;x-=root_joint[:,None,:]
            X.append(x.transpose(2,0,1));rows.append(dict(patient=r['subject'],trial=r['trial'],activity=r['activity'],file=r['file'],row_id=f"{r['file']}:{end}",frame_end=end,y=int(r['y'][end-1])))
    X=np.stack(X);d=pd.DataFrame(rows)
    d.to_csv(Path(out,'movement_windows.csv'),index=False)
    # A derived skeleton identity attack is not a raw-video comparison.
    return X,d,audit

def summaries(X):
    return np.concatenate([X.mean(axis=2).reshape(len(X),-1),X.std(axis=2).reshape(len(X),-1),np.abs(np.diff(X,axis=2)).mean(axis=2).reshape(len(X),-1)],axis=1)

def train_graph(X,y,A,cfg,seed,tag,out):
    torch.manual_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=STGCN(A,tuple(cfg['channels']));optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    generator=torch.Generator().manual_seed(seed)
    data=torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y,dtype=torch.float32))
    loader=torch.utils.data.DataLoader(data,batch_size=cfg['batch_size'],shuffle=True,generator=generator)
    trace=[]
    for epoch in range(cfg['epochs']):
        model.train();losses=[]
        for bx,by in loader:
            optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(bx),by);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        trace.append(dict(model=tag,epoch=epoch+1,loss=np.mean(losses)))
        if (epoch+1)%5==0:print('MOVEMENT_TRAIN',json.dumps(trace[-1]),flush=True)
    pd.DataFrame(trace).to_csv(Path(out,'models',tag+'-training.csv'),index=False)
    torch.save(model.state_dict(),Path(out,'models',tag+'.pt'));model.eval()
    return model

def predict_graph(model,X):
    batches=np.array_split(X,max(1,int(np.ceil(len(X)/128))))
    with torch.no_grad():return torch.cat([torch.sigmoid(model(torch.tensor(x))) for x in batches]).numpy()

def evaluate(root,cfg,out):
    Path(out,'models').mkdir(exist_ok=True)
    X,d,audit=windows(root,out);F=summaries(X);pred=[];calibrations=[];splits=[];rows=[]
    units=sorted(d.patient.unique())
    for seed in cfg['seeds']:
        for i,held in enumerate(units):
            calunit=units[(i+1)%len(units)];train=~d.patient.isin([held,calunit]);cal=d.patient==calunit;test=d.patient==held
            splits.append(dict(seed=seed,test=int(held),calibration=int(calunit),train=[int(x) for x in units if x not in [held,calunit]]))
            for name in ['summary','stgcn','topology_free','shuffled']:
                if name=='summary':
                    model=fit_model(F[train],d.y[train]);cs=probability(model,F[cal]);ts=probability(model,F[test])
                else:
                    # Train-only scale, common to every graph ablation.
                    scale=np.std(X[train],axis=(0,2,3),keepdims=True);scale=np.maximum(scale,1e-6)
                    xx=X/scale
                    model=train_graph(xx[train],d.y[train].to_numpy(),adjacency(name,seed),cfg['movement'],seed,f'{name}-seed{seed}-test{held}',out)
                    cs=predict_graph(model,xx[cal]);ts=predict_graph(model,xx[test])
                crc=calibrate(d.y[cal],cs,d.patient[cal],cfg['alpha']);crc.pop('curve')
                calibrations.append(dict(seed=seed,heldout=int(held),model=name,**crc))
                z=d[test].copy();z['score']=ts;z['lambda']=crc['lambda'];z['seed']=seed;z['heldout']=held;z['model']=name;z['source']='OPEN DERIVED SKELETONS';pred.append(z)
                for policy,lam in [('fixed_0.5',.5),('crc',crc['lambda'])]:
                    row=dict(seed=seed,heldout=int(held),model=name,policy=policy,**metrics(z.y,z.score,z.patient,lam));rows.append(row)
                print('MOVEMENT_TEST',json.dumps(rows[-2],default=float),flush=True)
    predictions=pd.concat(pred,ignore_index=True);predictions.to_csv(Path(out,'movement_predictions.csv'),index=False)
    pd.DataFrame(rows).to_csv(Path(out,'movement_fold_metrics.csv'),index=False);pd.DataFrame(calibrations).to_csv(Path(out,'movement_calibration.csv'),index=False)
    Path(out,'movement_splits.json').write_text(json.dumps(splits,indent=2))
    aggregate=[];intervals=[];deltas=[]
    for (seed,name),z in predictions.groupby(['seed','model']):
        for subset,zz in [('all_heldout',z),('trial3_stress',z[z.trial==3])]:
            if not len(zz):continue
            aggregate.append(dict(seed=seed,model=name,subset=subset,policy='fixed_0.5',**metrics(zz.y,zz.score,zz.patient,.5)))
        if seed==cfg['seeds'][0]:
            q=z.copy();q['lambda']=.5
            intervals.extend(dict(model=name,**v) for v in cluster_bootstrap(q,seed,cfg['bootstrap_replicates']))
    for name in ['summary','topology_free','shuffled']:
        a=predictions[(predictions.seed==cfg['seeds'][0])&(predictions.model=='stgcn')]
        b=predictions[(predictions.seed==cfg['seeds'][0])&(predictions.model==name)]
        deltas.extend(dict(model_a='stgcn',model_b=name,**v) for v in paired_bootstrap(a,b,cfg['seeds'][0],cfg['bootstrap_replicates']))
    pd.DataFrame(aggregate).to_csv(Path(out,'movement_metrics.csv'),index=False);pd.DataFrame(intervals).to_csv(Path(out,'movement_intervals.csv'),index=False);pd.DataFrame(deltas).to_csv(Path(out,'movement_paired_effects.csv'),index=False)
    print('MOVEMENT_SUMMARY',json.dumps(aggregate,default=float),flush=True)
    return predictions,pd.DataFrame(aggregate)
