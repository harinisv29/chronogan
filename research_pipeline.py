"""Sequential, resumable study execution. Run after full GAN training completes."""
import argparse
import json
import subprocess
import sys
import time
import yaml
from download_dataset import ROOT
from train import load_checkpoint
from utils.experiment import atomic_json
from utils.research import STUDY

def launch(config_path,log):
    config=yaml.safe_load(config_path.read_text()); folder=ROOT/config['run_dir']
    if (folder/'results'/'training.png').exists(): return
    args=[sys.executable,str(ROOT/'run.py'),'train.py','--config',str(config_path),'--device','cuda']
    if (folder/'checkpoints'/'last.pt').exists(): args+=['--resume',str(folder/'checkpoints'/'last.pt')]
    with log.open('a') as stream: subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,cwd=ROOT,check=True)

def main():
    assert (STUDY/'results'/'training.png').exists(),'Complete the main GAN run before the remaining study'
    from research_audit import audit
    audit()
    main_config=yaml.safe_load((ROOT/'config.full-training.yaml').read_text())
    for name,changes in [('no_mask',{'use_mask':False}),('no_delta',{'use_delta':False}),('no_adversarial',{'adversarial':0.})]:
        folder=STUDY/'ablations'/name; folder.mkdir(parents=True,exist_ok=True)
        config={**main_config,**changes,'run_dir':str(folder.relative_to(ROOT))}
        file=folder/'config.yaml'
        if not file.exists(): file.write_text(yaml.safe_dump(config,sort_keys=False))
        atomic_json(STUDY/'results'/'study-progress.json',{'stage':'ablation_training','variant':name})
        print('Starting ablation',name,flush=True); launch(file,folder/'training.log')
    state=load_checkpoint(STUDY/'models'/'chronogan_best.pt')
    from research_fit import train_model,evaluate_sepsis
    atomic_json(STUDY/'results'/'study-progress.json',{'stage':'baseline_training'})
    train_model('imputer',state)
    atomic_json(STUDY/'results'/'study-progress.json',{'stage':'baseline_training','method':'brits'})
    train_model('brits',state)
    for method in ['direct','mean','forward','chronogan']:
        atomic_json(STUDY/'results'/'study-progress.json',{'stage':'sepsis_training','method':method})
        train_model('sepsis',state,method)
    # Only now expose final test masks and labels to the selected models.
    atomic_json(STUDY/'results'/'study-progress.json',{'stage':'held_out_reconstruction_evaluation'})
    from research_evaluate import evaluate
    evaluate()
    atomic_json(STUDY/'results'/'study-progress.json',{'stage':'held_out_sepsis_evaluation'})
    evaluate_sepsis(state)
    atomic_json(STUDY/'results'/'study-progress.json',{'stage':'experiments_complete'})
    from research_finalize import finalize
    finalize()

if __name__=='__main__': main()
