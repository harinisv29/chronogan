"""Record facts from existing artifacts; never fill in missing experiment results."""
import json
import torch
from download_dataset import ROOT
from train import load_checkpoint

state=load_checkpoint(ROOT/'models'/'chronogan_best.pt')
summary=json.loads((ROOT/'data'/'summary.json').read_text())
evaluation=json.loads((ROOT/'results'/'evaluation.json').read_text())
sepsis=json.loads((ROOT/'results'/'sepsis.json').read_text())
inference=json.loads((ROOT/'results'/'inference.json').read_text())
lines=['# Actual workspace verification','',f'- Real records downloaded: {summary["patients"]} ({summary["sets"]}); hourly rows: {summary["observations"]}.',
       '- The official ZIP URLs returned 404. Original patient PSV files were downloaded from the working official PhysioNet training directories.',
       f'- Patient split: { {k:len(v) for k,v in state["splits"].items()} }.',
       f'- Training epochs completed: {len(state["history"])}; best validation normalized MAE: {state["best"]:.6f}.',
       f'- PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}.',
       f'- Checkpoint SHA-256: `{inference["checkpoint_sha256"]}`.',
       '- Automated validation: 11 tests passed, including held-out real-patient inference and web upload/reconstruction/CSV/mask/uncertainty/metadata/PDF exports.',
       '- Native observed clinical values are preserved; the held-out export is in `reconstructed.csv` with imputation flags.',
       '- Full 40,336-patient download and full training have NOT been run. This 200-patient, two-epoch cohort is a functionality smoke experiment, not a clinical validation.',
       '', '## Reconstruction at 20% random hiding','', '| Method | Normalized MAE | Normalized RMSE | Scored cells |','|---|---:|---:|---:|']
for row in evaluation['experiments']:
    if row['rate']==.2 and row['pattern']=='random':
        lines.append(f'| {row["method"]} | {row["normalized_mae"]:.6f} | {row["normalized_rmse"]:.6f} | {row["held_out_cells"]} |')
lines+=['','## Downstream smoke experiments','','| Method | AUROC | AUPRC | Brier |','|---|---:|---:|---:|']
for method,metrics in sepsis['methods'].items():
    def fmt(key): return f'{metrics[key]:.6f}' if metrics.get(key) is not None else 'Unavailable'
    lines.append(f'| {method} | {fmt("auroc")} | {fmt("auprc")} | {fmt("brier")} |')
lines+=['','All methods use the same patient split. These small results do not establish an improvement over simpler methods. Linear interpolation is retrospective and excluded from prospective sepsis experiments.',
        '', 'Training, comparison, robustness, calibration, and held-out reconstruction PNGs are available in this directory. Noise intervals are model sample variability, not validated clinical confidence intervals.',
        '', 'Research estimates must not be treated as verified clinical observations.']
(ROOT/'results'/'verification.md').write_text('\n'.join(lines),encoding='utf-8')
print('\n'.join(lines))
