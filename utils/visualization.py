import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from download_dataset import ROOT

def training_plot(history,path=None):
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in ['generator','discriminator']:
        axes[0].plot([r['epoch'] for r in history],[r[name] for r in history],label=name)
    axes[0].legend(); axes[0].set_title('Training objectives')
    axes[1].plot([r['epoch'] for r in history],[r['validation_normalized_mae'] for r in history])
    axes[1].set_title('Validation MAE (normalized)')
    fig.tight_layout(); fig.savefig(path or ROOT/'results'/'training.png'); plt.close(fig)

def comparison_plot(rows,folder=None):
    folder=folder or ROOT/'results'
    selected=[r for r in rows if r['rate']==.2 and r['pattern']=='random']
    fig,ax=plt.subplots(figsize=(9,4))
    ax.bar([r['method'] for r in selected],[r['normalized_mae'] for r in selected])
    ax.set_ylabel('Normalized MAE'); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(folder/'comparison.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,pattern in zip(axes,['random','block']):
        for method in sorted({r['method'] for r in rows}):
            values=[r for r in rows if r['method']==method and r['pattern']==pattern]
            ax.plot([r['rate'] for r in values],[r['normalized_mae'] for r in values],marker='o',label=method)
        ax.set_title(pattern+' masking'); ax.set_xlabel('Artificial hiding rate'); ax.set_ylabel('Normalized MAE')
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(folder/'robustness.png'); plt.close(fig)

def calibration_plot(methods,folder=None):
    folder=folder or ROOT/'results'
    fig,ax=plt.subplots(figsize=(6,5)); ax.plot([0,1],[0,1],'--',color='gray')
    for name,result in methods.items():
        if 'calibration' in result:
            ax.plot(result['calibration']['predicted'],result['calibration']['observed'],marker='o',label=name)
    ax.set_xlabel('Predicted probability'); ax.set_ylabel('Observed positive fraction'); ax.legend()
    fig.tight_layout(); fig.savefig(folder/'calibration.png'); plt.close(fig)
