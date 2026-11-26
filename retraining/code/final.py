import pandas as pd,numpy as np,json
exec(open('stage2b.py').read().split("def run(")[0])
v='clip125'; Xo,ya,fold,Xt,yt=load(v); yv=np.minimum(ya,125)
def met2(p,y):
    e=p-y; return dict(MAE=abs(e).mean(),RMSE=np.sqrt((e**2).mean()),R2=1-(e**2).sum()/((y-y.mean())**2).sum(),w10=(abs(e)<=10).mean()*100,w20=(abs(e)<=20).mean()*100,PHM=phm(e),bias=e.mean())
P={}; V={}
for i,m in enumerate(M): P[m]=Xt[:,i]; V[m]=Xo[:,i]
V['simple']=Xo.mean(1); P['simple']=Xt.mean(1)
oo=np.zeros(len(yv))
for f in np.unique(fold):
    mk=fold==f; w=fitw(Xo[~mk],yv[~mk],'mae'); oo[mk]=Xo[mk]@w
w=fitw(Xo,yv,'mae'); V['weighted']=oo; P['weighted']=Xt@w
from sklearn.linear_model import LinearRegression
oo=np.zeros(len(yv))
for f in np.unique(fold):
    mk=fold==f; s=LinearRegression(positive=True).fit(Xo[~mk],yv[~mk]); oo[mk]=s.predict(Xo[mk])
s=LinearRegression(positive=True).fit(Xo,yv); V['stacking']=oo; P['stacking']=s.predict(Xt)
rows={}
for k in P: rows[k]={**{'val_'+a:b for a,b in met2(V[k],yv).items()},**{'test_'+a:b for a,b in met2(P[k],yt).items()}}
R=pd.DataFrame(rows).T; R.to_csv('final_results.csv'); pd.set_option('display.width',250)
print(R[['val_MAE','val_RMSE','val_R2','test_MAE','test_RMSE','test_R2','test_w10','test_w20','test_PHM','test_bias']].round(3))
print('weights',dict(zip(M,w.round(3)))); print('stack',dict(zip(M,s.coef_.round(3))),s.intercept_)
# fold MAE
fm=pd.DataFrame({k:[abs(V[k][fold==f]-yv[fold==f]).mean() for f in np.unique(fold)] for k in ['weighted','lgbm_roll','xgb_roll','et_roll','xgb','stacking']}).round(2); print(fm); fm.to_csv('fold_mae.csv')
rng=np.random.default_rng(42)
for b in ['lgbm_roll','xgb_roll','et_roll']:
    d=abs(P['weighted']-yt)-abs(P[b]-yt); bs=[rng.choice(d,100).mean() for _ in range(5000)]; print('weighted-',b,round(d.mean(),3),np.percentile(bs,[2.5,97.5]).round(2),'wins',(d<0).sum(),(d>0).sum())
    d=(P['weighted']-yt)**2-(P[b]-yt)**2
for k in P: pd.DataFrame({'unit':np.arange(1,101),'actual':yt,'pred':P[k]}).to_csv(f'final_test_pred_{k}.csv',index=False)
np.save('final_w.npy',w)
