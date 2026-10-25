"""Isolated, sequential CUDA measurements after the protected trainer exits.

Never writes research checkpoints, splits, masks or optimizer configuration.
All throughput comparisons use real cached training windows, not test data.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from download_dataset import ROOT
from train import load_checkpoint
from models.generator import Generator
from models.discriminator import Discriminator
from utils.dataset import artificial_mask,recompute_delta
from utils.losses import masked_mean,generator_loss,prepare_physiological_bounds
from utils.performance import WorkerSafeWindows,BatchedWindows,collate_cached_batch,DeferredMetrics
from utils.experiment import atomic_json,atomic_save


class Telemetry:
    def __init__(self): self.rows=[]; self.stop=threading.Event()
    def __enter__(self):
        def poll():
            while not self.stop.is_set():
                result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                if result.returncode==0: self.rows.append([float(v) for v in result.stdout.strip().split(',')])
                self.stop.wait(.5)
        self.thread=threading.Thread(target=poll,daemon=True); self.thread.start(); return self
    def __exit__(self,*args): self.stop.set(); self.thread.join(timeout=12)
    def summary(self):
        return {'gpu_utilization_mean_percent':statistics.mean(r[0] for r in self.rows) if self.rows else None,
                'whole_device_peak_mib':max((r[1] for r in self.rows),default=None),'telemetry_samples':len(self.rows)}


def make_loader(folder,batch,workers=0,batched=False,pin=True,prefetch=2):
    dataset=(BatchedWindows if batched else WorkerSafeWindows)(folder)
    options={'batch_size':batch,'shuffle':False,'num_workers':workers,'pin_memory':pin}
    if batched: options['collate_fn']=collate_cached_batch
    if workers: options.update(persistent_workers=True,prefetch_factor=prefetch,timeout=90)
    return DataLoader(dataset,**options)


def close_loader(iterator):
    if hasattr(iterator,'_shutdown_workers'): iterator._shutdown_workers()


class Step:
    def __init__(self,state,optimized=False,fused=False):
        self.c=state['config']; self.stats=state['stats']; self.optimized=optimized
        self.g=Generator(34,self.c['hidden'],self.c['noise']).cuda()
        self.g.use_mask=self.c.get('use_mask',True); self.g.use_delta=self.c.get('use_delta',True)
        self.d=Discriminator(34,self.c['hidden']).cuda()
        self.g.load_state_dict(state['generator']); self.d.load_state_dict(state['discriminator'])
        self.go=torch.optim.Adam(self.g.parameters(),lr=self.c['learning_rate'],fused=fused if fused else None)
        self.do=torch.optim.Adam(self.d.parameters(),lr=self.c['learning_rate'],fused=fused if fused else None)
        self.bounds=prepare_physiological_bounds(self.stats,'cuda') if optimized else None
        self.metrics=DeferredMetrics(); self.phases={}; self.events=False

    def section(self,name,function):
        with torch.profiler.record_function(name):
            if not self.events: return function()
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(); clock=time.perf_counter(); result=function(); end.record()
            self.phases.setdefault(name,[]).append((start,end,time.perf_counter()-clock))
            return result

    def __call__(self,batch):
        x,m,delta,v=self.section('host_to_cuda',lambda:tuple(a.cuda(non_blocking=True) for a in batch[:4]))
        def masks():
            visible,held=artificial_mask(m,v,self.c['hide_rate'])
            dd=recompute_delta(visible,delta); reveal=(torch.rand_like(m)<.9).float()
            return visible,held,dd,reveal*visible+.5*(1-reveal),1-reveal
        visible,held,dd,hint,unknown=self.section('mask_and_gap',masks)
        def forward():
            with torch.autocast('cuda',dtype=torch.bfloat16): return self.g(x*visible,visible,dd,v)
        out=self.section('generator_forward',forward)
        self.do.zero_grad(set_to_none=True)
        def discriminator_loss():
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=self.d(out.detach(),hint,v)
            return masked_mean(F.binary_cross_entropy_with_logits(logits.float(),visible,reduction='none'),unknown*v[...,None])
        dl=self.section('discriminator_forward_loss',discriminator_loss)
        self.section('discriminator_finite_check',lambda:finite(dl))
        self.section('discriminator_backward',dl.backward)
        self.section('discriminator_clip',lambda:torch.nn.utils.clip_grad_norm_(self.d.parameters(),5,error_if_nonfinite=True))
        self.section('discriminator_optimizer',self.do.step)
        for p in self.d.parameters(): p.requires_grad_(False)
        self.go.zero_grad(set_to_none=True)
        def generator_objective():
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=self.d(out,hint,v)
            return generator_loss(out,x,held,visible,v,logits.float(),unknown,self.c,self.stats,True,prepared_bounds=self.bounds)
        gl,parts=self.section('generator_discriminator_and_loss',generator_objective)
        self.section('generator_finite_check',lambda:finite(gl))
        self.section('generator_backward',gl.backward)
        self.section('generator_clip',lambda:torch.nn.utils.clip_grad_norm_(self.g.parameters(),5,error_if_nonfinite=True))
        self.section('generator_optimizer',self.go.step)
        for p in self.d.parameters(): p.requires_grad_(True)
        def logging():
            if self.optimized: self.metrics.add(gl,dl,parts)
            else: return [gl.item(),dl.item(),*[float(value.detach()) for value in parts.values()]]
        self.section('batch_metrics_cpu_sync',logging)
        return gl.detach(),dl.detach()


def finite(value):
    if not torch.isfinite(value): raise FloatingPointError('Nonfinite benchmark loss')


def rng_state(): return torch.get_rng_state(),torch.cuda.get_rng_state_all()
def set_rng(state): torch.set_rng_state(state[0]); torch.cuda.set_rng_state_all(state[1])


def correctness(state,batch):
    reference=Step(state); candidate=Step(state,True); torch.manual_seed(42); rng=rng_state()
    a=reference(batch); set_rng(rng); b=candidate(batch)
    result={}
    for label,first,second in [('generator_loss',a[0],b[0]),('discriminator_loss',a[1],b[1])]:
        result[label+'_bitwise_equal']=torch.equal(first,second)
    for name in ['g','d']:
        result[name+'_parameters_bitwise_equal']=all(torch.equal(a,b) for a,b in zip(getattr(reference,name).parameters(),getattr(candidate,name).parameters()))
        result[name+'_gradients_bitwise_equal']=all((a.grad is None and b.grad is None) or
            (a.grad is not None and b.grad is not None and torch.equal(a.grad,b.grad))
            for a,b in zip(getattr(reference,name).parameters(),getattr(candidate,name).parameters()))
    result['passed']=all(result.values())
    del reference,candidate; gc.collect(); torch.cuda.empty_cache()
    return result


def throughput(state,folder,batch,optimized=False,fused=False,iterations=30):
    loader=make_loader(folder,64,batched=optimized); iterator=iter(loader)
    original=next(iterator)
    # Identical real 64-window content across sizes; repeat only the batch axis.
    data=tuple(value.repeat((batch//64,)+(1,)*(value.ndim-1)).pin_memory() for value in original)
    step=Step(state,optimized,fused); torch.manual_seed(42)
    for _ in range(6): step(data)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); samples=0; durations=[]
    with Telemetry() as telemetry:
        started=time.perf_counter()
        for _ in range(iterations):
            before=torch.cuda.Event(enable_timing=True); after=torch.cuda.Event(enable_timing=True)
            before.record(); step(data); after.record(); durations.append((before,after)); samples+=len(data[0])
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
    result={'batch_size':batch,'optimized':optimized,'fused_adam':fused,'iterations':iterations,'warmup_iterations':6,
            'samples_per_second':samples/elapsed,'batches_per_second':iterations/elapsed,
            'wall_step_ms':elapsed/iterations*1000,'cuda_stream_step_ms':statistics.mean(a.elapsed_time(b) for a,b in durations),
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'finite_losses':True,**telemetry.summary()}
    result['inputs']='Same real 64-window content repeated along batch dimension; host batching measured separately by loader and epoch replays.'
    close_loader(iterator); del step,iterator,loader; gc.collect(); torch.cuda.empty_cache()
    return result


def loader_bench(folder):
    rows=[]
    for workers in [0,2,4,8]:
        for pin in ([False,True] if workers==0 else [True]):
            for prefetch in ([2,4] if workers in (2,4) else [2]):
                iterator=None
                try:
                    loader=make_loader(folder,64,workers,pin=pin,prefetch=prefetch)
                    clock=time.perf_counter(); iterator=iter(loader); first=next(iterator); startup=time.perf_counter()-clock
                    for _ in range(4): next(iterator)
                    clock=time.perf_counter()
                    for _ in range(40): next(iterator)
                    elapsed=time.perf_counter()-clock
                    rows.append({'workers':workers,'pin_memory':pin,'persistent_workers':bool(workers),'prefetch_factor':prefetch if workers else None,'startup_seconds':startup,'steady_ms_per_batch':elapsed/40*1000})
                except Exception as error:
                    rows.append({'workers':workers,'pin_memory':pin,'error':str(error)})
                finally:
                    if iterator is not None: close_loader(iterator)
    # Vectorized loading is tested independently; future training stays at workers=0
    # unless a separately validated worker recommendation is explicitly produced.
    loader=make_loader(folder,64,batched=True); iterator=iter(loader)
    for _ in range(5): next(iterator)
    clock=time.perf_counter()
    for _ in range(40): next(iterator)
    rows.append({'workers':0,'batched':True,'pin_memory':True,'steady_ms_per_batch':(time.perf_counter()-clock)/40*1000})
    close_loader(iterator)
    return rows


def profiler_run(state,folder,target):
    iterator=iter(make_loader(folder,64)); batch=next(iterator); step=Step(state)
    for _ in range(5): step(batch)
    torch.cuda.synchronize(); step.events=True
    activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(): activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True) as prof:
            for _ in range(8): step(batch); prof.step()
        torch.cuda.synchronize(); prof.export_chrome_trace(str(target/'trace.json'))
        (target/'profiler-cpu.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=35))
        (target/'profiler-cuda.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total',row_limit=35))
        operations=[{'name':event.key,'count':event.count,'self_cpu_us':event.self_cpu_time_total,'self_device_us':getattr(event,'self_device_time_total',0)} for event in prof.key_averages()]
        atomic_json(target/'profiler-operations.json',operations)
        profiler={'trace_saved':True,'requested_activities':[str(a) for a in activities]}
    except Exception as error:
        profiler={'trace_saved':False,'error':str(error)}
        step.events=False; step.phases={}
        for _ in range(5): step(batch)
        step.events=True
        for _ in range(8): step(batch)
        torch.cuda.synchronize()
    phases={name:{'host_ms':statistics.mean(row[2] for row in rows)*1000,
                  'cuda_stream_elapsed_ms':statistics.mean(a.elapsed_time(b) for a,b,_ in rows)} for name,rows in step.phases.items()}
    close_loader(iterator); del step,iterator; gc.collect(); torch.cuda.empty_cache()
    return {'profiler':profiler,'phases':phases,'warning':'Profiler adds overhead. CUDA-event intervals include launch gaps/synchronization; kernel activity is separate in trace if CUPTI is available.'}


def epoch_replay(state,folder,val_folder,optimized,target):
    """Matched full training+validation replay; discarded benchmark weights."""
    loader=make_loader(folder,64,batched=optimized); val_loader=make_loader(val_folder,64,batched=optimized)
    step=Step(state,optimized); torch.manual_seed(42)
    # Warm kernels on a separate disposable model, keeping replay initialization fixed.
    warm=Step(state,optimized); batch=next(iter(loader))
    for _ in range(5): warm(batch)
    del warm; gc.collect(); torch.cuda.synchronize(); torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(); count=0; data_seconds=0.
    with Telemetry() as telemetry:
        clock=time.perf_counter(); iterator=iter(loader)
        while True:
            start=time.perf_counter()
            try: batch=next(iterator)
            except StopIteration: break
            data_seconds+=time.perf_counter()-start; step(batch); count+=len(batch[0])
        if optimized: step.metrics.means()
        train_clock=time.perf_counter(); step.g.eval(); error=0.; scored=0.
        mask_rng=torch.Generator(device='cuda').manual_seed(12345); noise_rng=torch.Generator(device='cuda').manual_seed(54321)
        with torch.no_grad():
            for batch in val_loader:
                x,m,d,v=[a.cuda(non_blocking=optimized) for a in batch[:4]]
                visible,held=artificial_mask(m,v,state['config']['hide_rate'],generator=mask_rng)
                z=torch.randn(*x.shape[:2],step.g.noise,device='cuda',generator=noise_rng)
                with torch.autocast('cuda',dtype=torch.bfloat16): output=step.g(x*visible,visible,recompute_delta(visible,d),v,z)
                error+=((output-x).abs()*held).sum().item(); scored+=held.sum().item()
        torch.cuda.synchronize(); elapsed=time.perf_counter()-clock
    validation=error/scored
    # Preserve only benchmark timing/state evidence, never official experiment weights.
    result={'epoch_seconds':elapsed,'training_seconds':train_clock-clock,'validation_seconds':elapsed-(train_clock-clock),
            'samples_per_second_training':count/(train_clock-clock),'patients_used':'training/validation only',
            'training_windows':count,'cpu_loader_seconds':data_seconds,'validation_normalized_mae':validation,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,**telemetry.summary()}
    payload={**state,'generator':step.g.state_dict(),'discriminator':step.d.state_dict(),
             'generator_optimizer':step.go.state_dict(),'discriminator_optimizer':step.do.state_dict(),
             'benchmark_only_discarded_replay':True}
    clock=time.perf_counter(); atomic_save(target/('optimized-save-timing.pt' if optimized else 'reference-save-timing.pt'),payload)
    result['checkpoint_write_seconds']=time.perf_counter()-clock
    clock=time.perf_counter(); atomic_json(target/('optimized-logging-timing.json' if optimized else 'reference-logging-timing.json'),{'history':state['history'],'progress':{'stage':'benchmark_only'}})
    result['epoch_json_write_seconds']=time.perf_counter()-clock
    final={name:{k:v.detach().cpu().clone() for k,v in getattr(step,name).state_dict().items()} for name in ['g','d']}
    close_loader(iterator); del step,loader,val_loader; gc.collect(); torch.cuda.empty_cache()
    return result,final


def main(request_path):
    request=json.loads(Path(request_path).read_text()); target=ROOT/request['directory']; target.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    if os.getpid()==request['protected_pid']: raise RuntimeError('Cannot profile inside the protected running trainer')
    assert (ROOT/'runs/full-training/ablations/no_mask/results/training.png').exists(),'Protected experiment must finish before profiling'
    state=load_checkpoint(ROOT/request['checkpoint'])
    protected_hash=hashlib.sha256((ROOT/request['checkpoint']).read_bytes()).hexdigest()
    manifest=json.loads((target/'source-manifest.json').read_text())
    fixed={name:value for name,value in manifest.items() if name.startswith('data/') or name.startswith('runs/full-training/')}
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in fixed.items())
    folder=ROOT/state['config']['cache_dir']/'train'; val_folder=folder.parent/'val'
    assert torch.cuda.is_available()
    atomic_json(target/'status.json',{'stage':'gpu_profiling','pid':os.getpid(),'protected_pid':request['protected_pid'],'benchmark_only':True})
    iterator=iter(make_loader(folder,64)); batch=next(iterator)
    correctness_result=correctness(state,batch); close_loader(iterator)
    atomic_json(target/'correctness.json',correctness_result)
    phases=profiler_run(state,folder,target); atomic_json(target/'phases.json',phases)
    loaders=loader_bench(folder); atomic_json(target/'data-loaders.json',loaders)
    results=[]
    for size in [64,128,256,512]:
        try: result=throughput(state,folder,size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); result={'batch_size':size,'out_of_memory':True}
        results.append(result); atomic_json(target/'batch-sizes.json',results)
    optimized=throughput(state,folder,64,True)
    atomic_json(target/'optimized-steps.json',optimized)
    try: fused=throughput(state,folder,64,fused=True)
    except Exception as error: fused={'error':str(error)}
    atomic_json(target/'fused-adam.json',fused)
    before,reference_weights=epoch_replay(state,folder,val_folder,False,target)
    atomic_json(target/'epoch-reference.json',before)
    after,candidate_weights=epoch_replay(state,folder,val_folder,True,target)
    atomic_json(target/'epoch-optimized.json',after)
    epoch_equal=all(torch.equal(reference_weights[name][key],candidate_weights[name][key]) for name in reference_weights for key in reference_weights[name])
    correctness_result['full_epoch_parameters_bitwise_equal']=epoch_equal
    correctness_result['full_epoch_validation_identical']=before['validation_normalized_mae']==after['validation_normalized_mae']
    correctness_result['research_artifacts_unchanged']=all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in fixed.items()) and hashlib.sha256((ROOT/request['checkpoint']).read_bytes()).hexdigest()==protected_hash
    correctness_result['passed']=all(correctness_result.values()); atomic_json(target/'correctness.json',correctness_result)
    speedup=(1-after['epoch_seconds']/before['epoch_seconds'])*100
    apply=correctness_result['passed'] and speedup>=3
    recommendation={'performance_optimized':apply,'performance_source':str(target.relative_to(ROOT)),
                    'batch_size_unchanged':64,'workers_unchanged':0,'fused_adam_applied':False,'compile_applied':False}
    atomic_json(target/'recommendation.json',recommendation)
    history=json.loads((ROOT/'runs/full-training/ablations/no_mask/results/training.json').read_text())
    summary={'status':'complete','before':before,'after':after,'speedup_percent':speedup,'correctness':correctness_result,
             'historical_no_mask_epoch_seconds':sum(r['epoch_seconds'] for r in history)/len(history),
             'batch_benchmarks':results,'loader_benchmarks':loaders,'phases':phases,'recommendation':recommendation,
             'torch_compile':{'triton_installed':importlib.util.find_spec('triton') is not None,'applied':False,'reason':'Not enabled: Windows CUDA recurrent graph support and compile amortization have not been validated; no package/environment changes.'},
             'comparability':'Both replay epochs start with identical weights and fresh identical Adam optimizers, traverse the same training windows and fixed validation masks/noise. Larger batches and fused Adam are benchmark-only and require approval before scientific training. Original research runs were not restarted or modified.'}
    ranked=sorted(phases['phases'].items(),key=lambda row:row[1]['host_ms'],reverse=True)
    summary['measured_diagnostics']={'largest_host_regions':ranked[:5],
        'loader_fraction_of_training_wall_time':before['cpu_loader_seconds']/before['training_seconds'],
        'interpretation':'Host phase durations include any synchronization in that region; CUDA event intervals include launch gaps. Consult kernel activity in trace before attributing all elapsed stream time to active GPU computation.'}
    summary['cuda_execution']={'torch_version':str(torch.__version__),'cuda_runtime':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'cudnn_version':torch.backends.cudnn.version(),
        'cudnn_enabled':torch.backends.cudnn.enabled,'bf16_supported':torch.cuda.is_bf16_supported()}
    operation_file=target/'profiler-operations.json'
    if operation_file.exists():
        operations=json.loads(operation_file.read_text())
        summary['measured_diagnostics']['scalar_extractions_in_8_steps']=sum(row['count'] for row in operations if row['name']=='aten::_local_scalar_dense')
        summary['measured_diagnostics']['cudnn_rnn_calls_in_trace']=sum(row['count'] for row in operations if 'cudnn_rnn' in row['name'])
    summary['remaining_gan_time_projection']={'hours_for_two_50_epoch_runs_at_measured_replay_speed':after['epoch_seconds']*100/3600,
        'limitation':'Projection from no_mask replay only. Subsequent architectures, early stopping, baselines, downstream heads and evaluation have different durations; this is not a total-study completion guarantee.'}
    atomic_json(target/'report.json',summary)
    lines=['# ChronoGAN CUDA performance measurements','','Isolated benchmarks after the protected ablation completed; research checkpoints are unchanged. Full-epoch rows are matched benchmark replays, not new research experiments.','','| Metric | Reference | Optimized |','|---|---:|---:|',f'| Train + validation epoch (s) | {before["epoch_seconds"]:.3f} | {after["epoch_seconds"]:.3f} |',f'| Training windows/s | {before["samples_per_second_training"]:.2f} | {after["samples_per_second_training"]:.2f} |',f'| Mean GPU utilization (%) | {before["gpu_utilization_mean_percent"]} | {after["gpu_utilization_mean_percent"]} |',f'| Whole-device peak MiB | {before["whole_device_peak_mib"]} | {after["whole_device_peak_mib"]} |',f'| CPU loader time per epoch (s) | {before["cpu_loader_seconds"]:.3f} | {after["cpu_loader_seconds"]:.3f} |',f'| Checkpoint save (s) | {before["checkpoint_write_seconds"]:.4f} | {after["checkpoint_write_seconds"]:.4f} |','',f'Measured epoch time reduction: {speedup:.2f}%. Bitwise full-epoch parameter equality: {epoch_equal}. Same validation MAE: {correctness_result["full_epoch_validation_identical"]}. Safe optimized path enabled: {apply}.','',summary['comparability'],'','Detailed CUDA events, profiler operator counts, Chrome trace, loader workers 0/2/4/8, batch 64/128/256/512 and benchmark-only fused Adam are separate JSON/text artifacts. Profiling overhead is excluded from throughput/epoch comparisons. zero_grad already defaults to set_to_none=True in the installed optimizer; it is not credited as a new speedup. The existing GRU uses cuDNN where supported; the custom temporal encoder retains its mathematical recurrence. No compile or fused optimizer is deployed.']
    lines+=['','## Measured bottleneck evidence','',f'Cached loader waiting consumed {summary["measured_diagnostics"]["loader_fraction_of_training_wall_time"]:.2%} of reference training wall time. Largest instrumented host regions (ms/step): '+', '.join(f'{name}={values["host_ms"]:.3f}' for name,values in ranked[:5])+'.',summary['measured_diagnostics']['interpretation'],'','## CUDA step throughput','',f'Reference batch64 stream elapsed: {results[0].get("cuda_stream_step_ms")} ms; optimized batch64: {optimized["cuda_stream_step_ms"]:.3f} ms. These use CUDA events and synchronized measurement boundaries, with six warmup iterations and thirty timed iterations. Batch64 training wall throughput before/after: '+str(results[0].get('samples_per_second'))+' / '+str(optimized['samples_per_second'])+' samples/s. Larger batches are isolated benchmarks and do not alter the scientific study.']
    (target/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(target/'status.json',{'stage':'complete','safe_optimization_enabled':apply})
    print(json.dumps({'performance_report':str(target/'report.md'),'speedup_percent':speedup,'safe_optimization_enabled':apply}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True); args=parser.parse_args()
    main(args.request)
