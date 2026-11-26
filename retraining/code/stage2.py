import pandas as pd,numpy as np,sys
from scipy.optimize import minimize
M=['lr','rf','xgb','lgbm','et','mlp']
def load(v):
    O={m:pd.read_csv(f'oof_{v}_{m}.csv') for m in M}; T={m:pd.read_csv(f'test_{v}_{m}.csv').sort_values('unit') for m in M}
    base=O['xgb']; fold=base.fold.values; ya=base.actual.values
    # align raw rf order
    for m in M:
        O[m]=O[m].set_index(['unit','cycle']).loc[list(zip(base.unit,base.cycle))]
    Xo=np.column_stack([O[m].pred.values for m in M]); Xt=np.column_stack([T[m].pred.values for m in M]); yt=T['xgb'].actual.values
    return Xo,ya,fold,Xt,yt
def phm(d): return np.where(d<0,np.expm1(-d/13),np.expm1(d/10)).sum()
def met(p,y):
    e=p-y; return dict(MAE=abs(e).mean(),RMSE=np.sqrt((e**2).mean()),R2=1-(e**2).sum()/((y-y.mean())**2).sum(),w20=(abs(e)<=20).mean()*100,PHM=phm(e))
def fitw(X,y,loss):
    k=X.shape[1]
    f=(lambda w: abs(X@w-y).mean()) if loss=='mae' else (lambda w: ((X@w-y)**2).mean())
    best=None
    for s in range(3):
        w0=np.random.default_rng(s).dirichlet(np.ones(k)) if s else np.ones(k)/k
        r=minimize(f,w0,bounds=[(0,1)]*k,constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
        if best is None or r.fun<best.fun: best=r
    return best.x
def run(v,cap):
    Xo,ya,fold,Xt,yt=load(v); yv=np.minimum(ya,cap) if cap else ya
    out={}
    print('==',v,'validation target', 'capped 125' if cap else 'raw')
    for i,m in enumerate(M): out[m]=(met(Xo[:,i],yv),met(Xt[:,i],yt))
    Et=Xt-yt[:,None]; print('test err corr\n',pd.DataFrame(Et,columns=M).corr().round(2))
    out['simple_all']=(met(Xo.mean(1),yv),met(Xt.mean(1),yt))
    best_single=min(M,key=lambda m:out[m][0]['MAE'])
    # weighted (MAE) with nested CV estimate
    res={}
    for loss in ('mae','rmse'):
        oo=np.zeros(len(yv))
        for f in np.unique(fold):
            mk=fold==f; w=fitw(Xo[~mk],yv[~mk],loss); oo[mk]=Xo[mk]@w
        w=fitw(Xo,yv,loss); out[f'weighted_{loss}']=(met(oo,yv),met(Xt@w,yt)); res[loss]=w
        print('weights',loss,dict(zip(M,w.round(3))))
    # stacking: nonneg linear + intercept via nested CV
    from sklearn.linear_model import LinearRegression
    oo=np.zeros(len(yv))
    for f in np.unique(fold):
        mk=fold==f; s=LinearRegression(positive=True).fit(Xo[~mk],yv[~mk]); oo[mk]=s.predict(Xo[mk])
    s=LinearRegression(positive=True).fit(Xo,yv); out['stacking']=(met(oo,yv),met(s.predict(Xt),yt)); print('stack coef',dict(zip(M,s.coef_.round(3))),round(s.intercept_,2))
    t=pd.DataFrame({k:{**{'val_'+a:b for a,b in v_[0].items()},**{'test_'+a:b for a,b in v_[1].items()}} for k,v_ in out.items()}).T
    print(t[['val_MAE','val_RMSE','val_R2','test_MAE','test_RMSE','test_R2','test_w20','test_PHM']].round(3))
    t.to_csv(f'results_{v}.csv'); np.save(f'w_{v}.npy',res['mae'])
    print('best single by val MAE:',best_single)
run('raw',None); run('clip125',125)
