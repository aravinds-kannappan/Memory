from pathlib import Path
import json,copy
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import average_precision_score,roc_auc_score
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence,pad_sequence
from .revision_data import rows,preonset_rows

def tune_tabular(X,y,items,split,model,seed,out):
    tr=preonset_rows(items,split['train'],y);va=preonset_rows(items,split['validation'],y);best=None;best_ap=-1;records=[]
    if model=='logistic':settings=[dict(C=c,class_weight=w) for c in [.1,1.] for w in [None,'balanced']]
    else:settings=[dict(max_leaf_nodes=l,l2_regularization=r,class_weight=w) for l,r,w in [(15,1.,None),(31,5.,None),(31,5.,'balanced')]]
    for h in settings:
        if model=='logistic':est=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression(**h,max_iter=350,random_state=seed))
        else:est=make_pipeline(VarianceThreshold(),HistGradientBoostingClassifier(**h,max_iter=120,learning_rate=.08,early_stopping=False,random_state=seed))
        est.fit(X[tr],y[tr]);score=est.predict_proba(X[va])[:,1];ap=average_precision_score(y[va],score)
        rec=dict(model=model,seed=seed,hyperparameters=h,validation_ap=float(ap),validation_auc=float(roc_auc_score(y[va],score)));records.append(rec);print('TUNING',json.dumps(rec),flush=True)
        if ap>best_ap:best_ap=ap;best=est
    Path(out,f'tuning-{model}-{seed}.json').write_text(json.dumps(records,indent=2));return best

class GRU(nn.Module):
    def __init__(self,d,hidden):
        super().__init__();self.gru=nn.GRU(d,hidden,batch_first=True);self.head=nn.Linear(hidden,1)
    def forward(self,x,lengths):
        packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False);z,_=self.gru(packed);z,_=pad_packed_sequence(z,batch_first=True,total_length=x.shape[1]);return self.head(z).squeeze(-1)

class SequenceModel:
    def __init__(self,net,mean,scale):self.net=net;self.mean=mean;self.scale=scale
    def predict_sequences(self,X,items,ids,batch=64):
        self.net.eval();pred=[]
        with torch.no_grad():
            for start in range(0,len(ids),batch):
                seq=[torch.from_numpy(np.nan_to_num((X[items[i]['start']:items[i]['end']]-self.mean)/self.scale,nan=0.).clip(-10,10).astype('float32')) for i in ids[start:start+batch]]
                lengths=torch.tensor([len(x) for x in seq]);z=self.net(pad_sequence(seq,batch_first=True),lengths).sigmoid().numpy()
                pred.extend(z[j,:len(x)] for j,x in enumerate(seq))
        return np.concatenate(pred)

def tune_gru(X,y,items,split,seed,out,epochs=6):
    torch.set_num_threads(2);tr=preonset_rows(items,split['train'],y);va=preonset_rows(items,split['validation'],y)
    mean=np.nanmean(X[tr],axis=0);mean=np.nan_to_num(mean).astype('float32');scale=np.nanstd(X[tr],axis=0);scale=np.where(np.isfinite(scale)&(scale>1e-5),scale,1).astype('float32')
    Z=np.nan_to_num((X-mean)/scale,nan=0.).clip(-10,10).astype('float32');best=None;bestap=-1;history=[]
    for hidden,lr in [(32,.001),(64,.001)]:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);net=GRU(X.shape[1],hidden);opt=torch.optim.Adam(net.parameters(),lr=lr)
        weight=min(20.,float((1-y[tr].mean())/y[tr].mean()));criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight),reduction='none')
        for epoch in range(epochs):
            ids=rng.permutation(split['train']);net.train();losses=[]
            for start in range(0,len(ids),64):
                take=ids[start:start+64];xs=[torch.from_numpy(Z[items[i]['start']:items[i]['end']]) for i in take];ys=[torch.from_numpy(y[items[i]['start']:items[i]['end']].astype('float32')) for i in take]
                lengths=torch.tensor([len(x) for x in xs]);xx=pad_sequence(xs,batch_first=True);yy=pad_sequence(ys,batch_first=True);mask=torch.arange(xx.shape[1])[None,:]<lengths[:,None]
                for row,i in enumerate(take):
                    yyfull=y[items[i]['start']:items[i]['end']];pos=np.flatnonzero(yyfull)
                    if len(pos):mask[row,min(len(yyfull),int(pos[0]+6)):]=False
                opt.zero_grad();logits=net(xx,lengths);loss=criterion(logits,yy)[mask].mean();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();losses.append(loss.item())
            wrapper=SequenceModel(net,mean,scale);p=wrapper.predict_sequences(X,items,split['validation']);vrows=rows(items,split['validation']);keep=np.isin(vrows,va);ap=average_precision_score(y[va],p[keep])
            rec=dict(model='gru',seed=seed,hidden=hidden,epoch=epoch+1,train_loss=float(np.mean(losses)),validation_ap=float(ap));history.append(rec);print('GRU_TRAIN',json.dumps(rec),flush=True)
            if ap>bestap:bestap=ap;best=SequenceModel(copy.deepcopy(net),mean,scale)
    torch.save({'state_dict':best.net.state_dict(),'mean':mean,'scale':scale,'hidden':best.net.gru.hidden_size},Path(out,f'gru-{seed}.pt'));Path(out,f'tuning-gru-{seed}.json').write_text(json.dumps(history,indent=2));return best

def predict(est,X,items,ids):
    if isinstance(est,SequenceModel):return est.predict_sequences(X,items,ids)
    ix=rows(items,ids);return np.concatenate([est.predict_proba(X[ix[j:j+50000]])[:,1] for j in range(0,len(ix),50000)])
