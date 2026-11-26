import pandas as pd,numpy as np,sys,time,warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
D='/downloads/turbofan-rul-ml/'
cols=['unit','cycle']+[f's{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]
tr=pd.read_csv(D+'data/raw/train_FD001.txt',sep=r'\s+',header=None,names=cols)
te=pd.read_csv(D+'data/raw/test_FD001.txt',sep=r'\s+',header=None,names=cols)
rul=pd.read_csv(D+'data/raw/RUL_FD001.txt',header=None)[0].values
F=['cycle']+[f'sensor_{i}' for i in (2,3,4,7,8,9,11,12,13,14,15,17,20,21)]
tr['RUL']=tr.groupby('unit').cycle.transform('max')-tr.cycle
fa=pd.read_csv(D+'reports/hybrid/enhanced_models/fold_assignments.csv')
tr=tr.merge(fa.rename(columns={'engine_id':'unit','time_cycles':'cycle'}),on=['unit','cycle'],how='left'); assert tr.fold.notna().all()
last=te.groupby('unit').tail(1).sort_values('unit'); Xte=last[F].values
S=[c for c in F if c!='cycle']
def roll(df):
    g=df.groupby('unit')
    r=g[S].transform(lambda x:x.rolling(10,min_periods=1).mean()).add_suffix('_m10')
    d=df[S]-g[S].transform('first'); d=d.add_suffix('_d0')
    return pd.concat([df[F],r,d],axis=1)
trR=roll(tr); teR=roll(te); 
XteR=teR.loc[last.index].values
X=tr[F].values
def mk(name):
    if name in ('xgb_roll','et_roll','lr_roll','rf_roll'): return mk(name[:-5])
    if name=='lr': return make_pipeline(StandardScaler(),LinearRegression())
    if name=='rf': return (RandomForestRegressor(200,random_state=42,n_jobs=2) if variant=='raw' else RandomForestRegressor(150,min_samples_leaf=5,random_state=42,n_jobs=2))
    if name=='xgb': return XGBRegressor(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=2)
    if name=='lgbm': return LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,random_state=42,n_jobs=2,verbose=-1)
    if name=='et': return ExtraTreesRegressor(300,min_samples_leaf=5,random_state=42,n_jobs=2)
    if name=='mlp': return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=300,random_state=42))
    if name=='knn':
        from sklearn.neighbors import KNeighborsRegressor
        return make_pipeline(StandardScaler(),KNeighborsRegressor(50,weights='distance',n_jobs=2))
    if name=='lgbm_roll': return LGBMRegressor(n_estimators=400,learning_rate=.03,num_leaves=15,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.8,random_state=42,n_jobs=2,verbose=-1)
    if name=='mlp_roll': return make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=300,random_state=42))
    if name=='svr': return make_pipeline(StandardScaler(),SVR(C=50,epsilon=2.0,gamma='scale',cache_size=1000))
variant=sys.argv[1]; cap=None if variant=='raw' else 125
names=sys.argv[2].split(',')
y=tr.RUL.values.astype(float); yfit=y if cap is None else np.minimum(y,cap)
for nm in names:
    t0=time.time(); oof=np.zeros(len(tr)); XX,XT=(trR.values,XteR) if nm.endswith('_roll') else (X,Xte)
    for f in sorted(tr.fold.unique()):
        m=tr.fold.values==f; mod=mk(nm); mod.fit(XX[~m],yfit[~m]); oof[m]=mod.predict(XX[m])
    mod=mk(nm); mod.fit(XX,yfit); pt=mod.predict(XT)
    pd.DataFrame({'unit':tr.unit,'cycle':tr.cycle,'fold':tr.fold,'actual':y,'pred':oof}).to_csv(f'/downloads/ensemble_exp/oof_{variant}_{nm}.csv',index=False)
    pd.DataFrame({'unit':last.unit.values,'actual':rul,'pred':pt}).to_csv(f'/downloads/ensemble_exp/test_{variant}_{nm}.csv',index=False)
    e=pt-rul; print(variant,nm,'oofMAE %.2f'%abs(oof-y).mean(),'testMAE %.2f'%abs(e).mean(),'%.0fs'%(time.time()-t0),flush=True)
