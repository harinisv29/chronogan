import numpy as np
from research_weight_study import bootstrap,summarize,score

def test_patient_cluster_bootstrap_pairs_identical_predictions():
    # train,mask,patient,subset,metric,feature; valid synthetic sufficient stats.
    a=np.zeros((1,3,4,3,5,34))
    a[:,:,:,0,0]=1.;a[:,:,:,0,1]=1.;a[:,:,:,0,2]=1.;a[:,:,:,0,4]=1.
    a[:,:,:,1]=a[:,:,:,0]
    result=bootstrap(a,a,np.ones(34),iterations=50)
    assert result['mae_95_ci']==[1.,1.]
    assert result['paired_mae_difference_vs_original_95_ci']==[0.,0.]
    assert result['coverage_95_ci']==[1.,1.]
    summary=summarize(a,np.ones(34))
    assert summary['raw']['standardized_mae']==1.
    assert summary['raw']['standardized_rmse']==1.

def test_scoring_preserves_withholding_and_scores_only_hidden_truth():
    truth=np.ones((3,34));values=truth.copy();values[1]=np.nan;pred=truth.copy();pred[1,14]=-3.
    held=np.zeros_like(truth,dtype=bool);held[1]=True
    stats={'features':list(__import__('preprocess').FEATURES),'mean':[0.]*34,'std':[1.]*34,'count':[10]*34}
    scored=score([pred],[(truth,values,held,0,1)],stats,24)
    assert scored[0,0,2].sum()==34
    assert scored[0,1,2].sum()==33
    assert scored[0,1,3].sum()==0
    assert scored[0,2,2].sum()==1

def test_adaptation_seed_pairing_is_retained_in_control_contrasts():
    a=np.zeros((3,3,4,3,5,34))
    for seed in range(3):
        a[seed,:,:,0,0]=seed+1.;a[seed,:,:,0,1]=(seed+1.)**2
        a[seed,:,:,0,2]=1.;a[seed,:,:,0,4]=1.;a[seed,:,:,1]=a[seed,:,:,0]
    result=bootstrap(a,a,np.ones(34),iterations=50)
    assert result['paired_mae_difference_vs_original_95_ci']==[0.,0.]
