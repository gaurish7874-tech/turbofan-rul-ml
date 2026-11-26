import pandas as pd,numpy as np,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
BL='#2A4FA6'; CR='#E7DECB'; BG='#F4EFE4'; LB='#9FB4E8'; DK='#1c1c1c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.edgecolor':DK,'axes.labelcolor':DK,'xtick.color':DK,'ytick.color':DK})
W=pd.read_csv('final_test_pred_weighted.csv'); y=W.actual; p=W.pred
def save(fig,name,dpi=200): fig.savefig(name,dpi=dpi,facecolor=BG); plt.close(fig)
# fig1 actual vs predicted: shape 4087368 x 3557016 EMU
fig,ax=plt.subplots(figsize=(4087368/914400,3557016/914400)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
mx=max(y.max(),p.max())+5; xs=np.array([0,mx])
ax.fill_between(xs,xs-20,xs+20,color=LB,alpha=.35,label='±20 cycles'); ax.plot(xs,xs,color=DK,lw=1,label='perfect')
ax.scatter(y,p,s=28,color=BL,edgecolor=CR,lw=.5,zorder=3)
ax.set(xlim=(0,mx),ylim=(0,mx),xlabel='Actual RUL (cycles)',ylabel='Predicted RUL (cycles)',title='Weighted averaging · NASA test')
ax.legend(frameon=False,loc='upper left',fontsize=9); ax.grid(alpha=.25)
fig.tight_layout(); save(fig,'fig_avp.png')
# fig2 fold MAE: 3054096 x 2139696
fm=pd.read_csv('fold_mae.csv'); fig,ax=plt.subplots(figsize=(3054096/914400,2139696/914400)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
xx=np.arange(5); w=.27
for i,(c,l,col) in enumerate([('xgb','XGBoost (15 inputs)',CR),('lgbm_roll','LightGBM-H',LB),('weighted','Weighted avg.',BL)]):
    ax.bar(xx+(i-1)*w,fm[c],w,label=l,color=col,edgecolor=DK,lw=.4)
ax.set_xticks(xx); ax.set_xticklabels([f'F{i+1}' for i in xx]); ax.set_ylabel('MAE (cycles)',fontsize=9); ax.set_ylim(0,18)
ax.legend(frameon=False,fontsize=6.5,ncol=3,loc='upper center',columnspacing=.8,handlelength=1); ax.tick_params(labelsize=8); ax.grid(axis='y',alpha=.25)
fig.tight_layout(); save(fig,'fig_fold.png')
# fig3 error by RUL bucket: 4425696 x 2002536
b=pd.cut(y,[-1,50,100,1000]); fig,ax=plt.subplots(figsize=(4425696/914400,2002536/914400)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
for i,(k,l,col) in enumerate([('xgb','XGBoost (15 inputs)',CR),('lgbm_roll','LightGBM-H',LB),('weighted','Weighted avg.',BL)]):
    pk=pd.read_csv(f'final_test_pred_{k}.csv').pred; v=abs(pk-y).groupby(b,observed=True).mean().values
    ax.bar(np.arange(3)+(i-1)*.27,v,.27,label=l,color=col,edgecolor=DK,lw=.4)
ax.set_xticks(range(3)); ax.set_xticklabels(['RUL ≤ 50\n(33 engines)','51–100\n(34)','> 100\n(33)'],fontsize=8); ax.set_ylabel('MAE (cycles)',fontsize=9)
ax.set_ylim(0,22); ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper center',columnspacing=.8,handlelength=1); ax.tick_params(labelsize=8); ax.grid(axis='y',alpha=.25); fig.tight_layout(); save(fig,'fig_bucket.png')
# fig4 ablation: 5166360 x 1051560
fig,ax=plt.subplots(figsize=(5166360/914400,1051560/914400)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
cfg=['Raw target\n15 inputs','Capped 125\n15 inputs','Raw target\n+ history','Capped 125\n+ history']
xg=[18.81,13.79,17.02,10.04]; lg=[18.44,13.67,16.41,10.01]
xx=np.arange(4); ax.bar(xx-.2,xg,.4,color=BL,label='XGBoost'); ax.bar(xx+.2,lg,.4,color=LB,label='LightGBM',edgecolor=DK,lw=.3)
for a,b_ in zip(xx,xg): ax.text(a-.2,b_+.4,f'{b_:.1f}',ha='center',fontsize=7)
for a,b_ in zip(xx,lg): ax.text(a+.2,b_+.4,f'{b_:.1f}',ha='center',fontsize=7)
ax.set_xticks(xx); ax.set_xticklabels(cfg,fontsize=6.5); ax.set_ylim(0,24); ax.set_ylabel('Test MAE',fontsize=7); ax.tick_params(labelsize=6.5); ax.legend(frameon=False,fontsize=6.5,ncol=2,loc='upper right')
fig.tight_layout(pad=.3); save(fig,'fig_ablation.png',250)
