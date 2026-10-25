import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from sklearn.calibration import calibration_curve

def classification_metrics(y,p):
    y=np.asarray(y); p=np.asarray(p)
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    prob,truth=calibration_curve(y,p,n_bins=10,strategy='uniform')
    both=len(np.unique(y))==2
    return dict(auroc=float(roc_auc_score(y,p)) if both else None,auprc=float(average_precision_score(y,p)) if both else None,
                sensitivity=float(tp/max(tp+fn,1)),specificity=float(tn/max(tn+fp,1)),brier=float(brier_score_loss(y,p)),
                calibration={'observed':prob.tolist(),'predicted':truth.tolist()},threshold=.5,hours=len(y))
