"""One-shot between-experiment profiling, never an interruption of a live run."""
import json
import os
from pathlib import Path
import subprocess
import sys
from download_dataset import ROOT
from utils.experiment import atomic_json


def before_training(config,resume=None):
    request_path=ROOT/'runs'/'performance-request.json'
    if not request_path.exists(): return config
    request=json.loads(request_path.read_text())
    if os.getpid()==request['protected_pid']: return config
    run=Path(config.get('run_dir','.')).as_posix()
    if not run.startswith('runs/full-training/ablations/'): return config
    target=ROOT/request['directory']; target.mkdir(parents=True,exist_ok=True)
    done=target/'gate-complete.json'
    if not done.exists():
        if not (ROOT/'runs/full-training/ablations/no_mask/results/training.png').exists():
            raise RuntimeError('Protected run has no completion artifact; refusing to start profiling')
        # This hook is only read by the next CLI process. The protected process
        # already loaded its original module before this gate was installed.
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,request['protected_pid'])
        if handle:
            code=wintypes.DWORD()
            kernel.GetExitCodeProcess(handle,ctypes.byref(code))
            kernel.CloseHandle(handle)
            if code.value==259: raise RuntimeError('Protected trainer is still running; refusing competing profiling')
        lock=target/'gate.lock'
        try:
            descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError('Profiling gate already started; inspect its status before continuing')
        os.close(descriptor)
        progress=ROOT/'runs'/'full-training'/'results'/'study-progress.json'
        previous_progress=json.loads(progress.read_text()) if progress.exists() else None
        atomic_json(progress,{'stage':'performance_profiling','directory':request['directory'],'next_experiment':config['run_dir']})
        print('Protected experiment finished naturally. Running isolated performance measurements before the next experiment.',flush=True)
        try:
            with (target/'benchmark.log').open('a') as log:
                result=subprocess.run([sys.executable,str(ROOT/'run.py'),'performance_benchmark.py','--request',str(request_path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                atomic_json(target/'benchmark-failure.json',{'returncode':result.returncode,'fallback':'Original training execution retained; inspect benchmark.log'})
            atomic_json(done,{'returncode':result.returncode,'profiling_finished':True})
        except Exception as error:
            atomic_json(target/'benchmark-failure.json',{'error':str(error),'fallback':'Original training execution retained'})
            atomic_json(done,{'profiling_finished':False,'error':str(error)})
        finally:
            if previous_progress is not None: atomic_json(progress,previous_progress)
    recommendation=target/'recommendation.json'
    # Never change settings when resuming a previously started experiment.
    completed=json.loads(done.read_text())
    if resume or completed.get('returncode')!=0 or not recommendation.exists(): return config
    choice=json.loads(recommendation.read_text())
    if choice.get('performance_optimized'):
        config={**config,'performance_optimized':True,'performance_source':choice['performance_source']}
        run_path=ROOT/config['run_dir']; run_path.mkdir(parents=True,exist_ok=True)
        atomic_json(run_path/'execution-config.json',config)
    return config
