"""Export a reconstruction of an actual held-out patient after real training."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from download_dataset import ROOT
from train import load_checkpoint
from preprocess import read_patient
from predict import reconstruct

state=load_checkpoint(ROOT/'models'/'chronogan_best.pt')
patient=state['splits']['test'][0]
assert patient not in state['splits']['train']
frame=read_patient(patient)
result,mask,uncertainty,meta=reconstruct(frame,samples=20)
result.to_csv(ROOT/'results'/'reconstructed.csv',index=False)
mask.to_csv(ROOT/'results'/'mask.csv',index=False)
uncertainty.to_csv(ROOT/'results'/'uncertainty.csv',index=False)
meta['held_out_patient']=patient
(ROOT/'results'/'inference.json').write_text(json.dumps(meta,indent=2))
fig,axes=plt.subplots(3,1,figsize=(11,9))
for ax,feature in zip(axes,['HR','SBP','Temp']):
    time=np.arange(1,len(frame)+1)
    ax.plot(time,result[feature],color='#087f78',label='Reconstructed trajectory')
    ax.scatter(time,frame[feature],color='#18333d',label='Recorded',s=15,zorder=4)
    missing=mask[feature].to_numpy()
    ax.scatter(time[missing],result.loc[missing,feature],color='#cd972e',marker='D',s=15,label='Generated estimate')
    ax.fill_between(time,uncertainty[feature+'__p05'],uncertainty[feature+'__p95'],color='#087f78',alpha=.15,label='Noise quantiles 5–95%')
    ax.set_ylabel(feature); ax.legend(fontsize=7,loc='upper right')
axes[-1].set_xlabel('Hourly row'); fig.suptitle('Held-out real patient · research estimates, not verified observations')
fig.tight_layout(); fig.savefig(ROOT/'results'/'reconstruction.png'); plt.close(fig)
print(f'Exported {len(frame)} real held-out hourly rows; {int(mask.sum().sum())} generated estimates.')
