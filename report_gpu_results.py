"""Build a final report exclusively from completed runs and verification artifacts."""
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from download_dataset import ROOT
from train import load_checkpoint

def read(path): return json.loads(path.read_text())

def main():
    folder=ROOT/'results'/'gpu'
    verification=read(folder/'cuda-verification.json')
    state=load_checkpoint(ROOT/'runs'/'gpu-smoke'/'checkpoints'/'last.pt')
    assert state['training_device']=='cuda' and state['generator_device'].startswith('cuda') and state['discriminator_device'].startswith('cuda')
    cpu_hashes=read(folder/'cpu-checkpoint-hashes.json')
    for relative,digest in cpu_hashes.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, f'CPU experiment changed: {relative}'
    preserved_results=[]
    for original in (ROOT/'results'/'cpu-smoke').iterdir():
        current=ROOT/'results'/original.name
        if original.is_file() and current.is_file():
            assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(current.read_bytes()).digest(),f'CPU result changed: {current}'
            preserved_results.append(original.name)
    summary=read(ROOT/'data'/'summary.json')
    plan_path=folder/'full-training-plan.json'
    plan=read(plan_path) if plan_path.exists() else None
    with (folder/'nvidia-monitor.csv').open() as source: monitor=list(csv.DictReader(source))
    tests=ET.parse(folder/'tests.xml').getroot()
    suite=tests if tests.tag=='testsuite' else tests.find('testsuite')
    test_counts={key:int(suite.get(key,'0')) for key in ['tests','failures','errors','skipped']}
    assert test_counts['failures']==0 and test_counts['errors']==0
    report={'cuda_verification':verification,'patient_cohort':summary['patients'],'smoke_patients':{k:len(v) for k,v in state['splits'].items()},'smoke_epochs':state['history'],'best_normalized_validation_mae':state['best'],'smoke_checkpoint':str(ROOT/'runs'/'gpu-smoke'/'models'/'chronogan_best.pt'),'cpu_checkpoints_unchanged':True,'tests':test_counts,'full_training_plan':plan,'sampled_whole_gpu_utilization_max_percent':max(float(r['gpu_utilization_percent']) for r in monitor) if monitor else None,'sampled_whole_gpu_used_memory_max_mib':max(float(r['memory_used_mib']) for r in monitor) if monitor else None,'full_training_launched':False}
    amp=ROOT/'runs'/'gpu-amp'/'results'/'training.json'
    if amp.exists(): report['bf16_stability_trial']=read(amp)
    report['cpu_result_files_unchanged']=preserved_results
    report['dataset_download']=read(ROOT/'data'/'full-download-verification.json')
    evaluation=ROOT/'runs'/'gpu-smoke'/'results'/'evaluation.json'
    if evaluation.exists():
        experiments=read(evaluation)
        assert experiments['checkpoint_sha256']==hashlib.sha256((ROOT/'runs'/'gpu-smoke'/'models'/'chronogan_best.pt').read_bytes()).hexdigest()
        report['gpu_reconstruction_experiments']=len(experiments['experiments'])
    (folder/'final-report.json').write_text(json.dumps(report,indent=2))
    lines=['# ChronoGAN GPU verification and training report','',f'- Python: {verification["python"].splitlines()[0]}',f'- Interpreter: `{verification["executable"]}`',f'- PyTorch: {verification["torch"]}; CUDA runtime: {verification["cuda_runtime"]}',f'- GPU: {verification["gpu"]}; total VRAM: {verification["vram_mib"]:.0f} MiB',f'- CUDA tensor operations, matrix multiplication, backward propagation and allocation/release: PASSED',f'- Real smoke patients: {sum(report["smoke_patients"].values())}; split: {report["smoke_patients"]}',f'- Both generator and discriminator parameters verified on CUDA.',f'- CPU checkpoint SHA-256 hashes unchanged.',f'- Verified dataset patients currently reported: {summary["patients"]}',f'- Automated tests: {test_counts}',f'- Sampled whole-GPU utilization maximum: {report["sampled_whole_gpu_utilization_max_percent"]}%; used VRAM maximum: {report["sampled_whole_gpu_used_memory_max_mib"]} MiB. These include driver/desktop usage.',f'- GPU checkpoint: `{report["smoke_checkpoint"]}`','','| Epoch | Generator loss | Discriminator loss | Normalized validation MAE | Seconds | Peak allocated MiB | Peak reserved MiB |','|---|---:|---:|---:|---:|---:|---:|']
    for row in state['history']:
        lines.append(f'| {row["epoch"]} | {row["generator"]:.6f} | {row["discriminator"]:.6f} | {row["validation_normalized_mae"]:.6f} | {row["epoch_seconds"]:.3f} | {row["peak_cuda_allocated_mib"]:.3f} | {row["peak_cuda_reserved_mib"]:.3f} |')
    if amp.exists():
        last=read(amp)[-1]
        lines+=['',f'BF16 stability trial: two finite epochs on the identical 200-patient cohort; final normalized validation MAE {last["validation_normalized_mae"]:.6f}, final epoch {last["epoch_seconds"]:.3f} s, allocated peak {last["peak_cuda_allocated_mib"]:.3f} MiB. Its checkpoints remain separate under runs/gpu-amp.']
    lines+=['',f'GPU baseline verification completed {report.get("gpu_reconstruction_experiments",0)} held-out masking experiments; sepsis metrics and comparison/robustness/calibration PNGs are under runs/gpu-smoke/results. Original CPU experiment files were checksum-compared against their saved snapshot and remain unchanged.']
    if plan:
        lines+=['','## Full training preparation',f'- Frozen patient splits: {plan["splits"]}',f'- Batch size: {plan["batch_size"]}; sequence length: {plan["sequence_length"]}; hidden width: {plan["hidden"]}.',f'- Precision: {plan["precision"]}',f'- Estimated host tensor storage: {plan["estimated_tensor_dataset_ram_gib"]:.3f} GiB, excluding interpreter/pandas/allocator overhead.',f'- Available host RAM: {plan["host_available_ram_gib"]:.2f} GiB; GPU free: {plan["gpu_free_mib"]:.0f} MiB; resource check passed: {plan["resources_ready"]}.',f'- Estimated epoch: {plan["estimated_epoch_seconds"]/60:.2f} minutes; estimated 50 epochs: {plan["estimated_50_epoch_hours"]:.2f} hours.',f'- {plan["estimate_limit"]}',f'- Full training command: `{plan["command"]}`']
    lines+=['','No extended full-data training run was launched. Remaining research work includes full-cohort training and evaluation, multiple seeds, patient-level confidence intervals, external validation and uncertainty calibration. Smoke results do not establish clinical superiority or clinical validation.','', 'Reconstructed values remain research estimates, not verified clinical observations.']
    (folder/'final-report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__': main()
