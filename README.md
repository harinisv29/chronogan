# ChronoGAN

A PyTorch and Flask research application for missing clinical time-series reconstruction. All code lives in this new directory; the IDE extension configuration is untouched.

**Reconstructed values are research estimates, not verified clinical observations.** Stochastic intervals reflect the trained generator's noise sensitivity; they are not clinically validated confidence intervals.

## Install and launch (Windows PowerShell)

Use Python 3.12 or newer with a working pip. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py download_dataset.py
python run.py train.py --config config.quick.yaml
python run.py evaluate.py --baseline-epochs 2
python run.py test.py
python run.py smoke_test.py
python run.py app.py
```

Open http://127.0.0.1:5000. Upload a real held-out patient `.psv` from the checkpoint's `splits.test` list. The app refuses reconstruction without trained weights. Uploads are session-isolated temporary files removed when the process exits. Files may be CSV or pipe-separated PSV with consecutive hourly rows. At least one supported clinical feature is required; absent feature columns become missing inputs. Preview is read only. Exports preserve extra uploaded columns, original clinical observations, and individual imputation flags.

This workspace now has a dedicated embedded Python in `.runtime/python/python.exe`, copied from the original shared runtime without changing SugarSense. Dependencies installed locally into `.packages` can be used with:

```powershell
& '.\.runtime\python\python.exe' run.py test.py
& '.\.runtime\python\python.exe' run.py app.py
```

The `run.py` launcher adds local packages without changing another project's Python environment. A regular virtual environment is preferred on another computer. CUDA is selected automatically if your installed PyTorch build supports your GPU; the default installed build may be CPU-only. Use a compatible CUDA PyTorch build for GPU training.

### GPU training in this workspace

The launcher prefers the CUDA-enabled PyTorch package in `.cuda-packages` when present, and loads other dependencies from `.packages`. This keeps the original CPU installation available. The CUDA runtime is bundled with the official PyTorch Windows wheel; a compatible NVIDIA driver is required. The installer uses one pip process, extended timeouts, retries, project-local cache, and the stable PyTorch 2.11.0 CUDA 13.0 Windows/Python 3.12 wheel. The complete wheel must match the official SHA-256 before the reusable installer installs it. [Official build commands](https://pytorch.org/get-started/previous-versions/) list this supported release/index. Check GPU execution, then run the isolated smoke experiment:

```powershell
& '.\.runtime\python\python.exe' run.py install_gpu.py
& '.\.runtime\python\python.exe' run.py check_gpu.py
& '.\.runtime\python\python.exe' run.py gpu_smoke_runner.py
```

The smoke runner explicitly passes `--device cuda`; both networks' parameter devices are asserted. Its 200-patient, two-epoch run uses hidden size 64, noise size 16, a 64-window batch and FP32. GPU weights and histories are saved in `runs/gpu-smoke/`, independently of CPU files. Epoch times, peak allocated/reserved CUDA memory, validation MAE and losses are saved with the history. `results/gpu/` contains the hardware audit, computation/allocation/release checks, `nvidia-smi` monitor samples and CPU checkpoint hashes. Whole-GPU NVIDIA memory figures include desktop applications and the driver context; PyTorch allocator figures are process-specific.

`requirements-gpu-lock.txt` records the separate GPU environment. Setting `CHRONOGAN_CPU_ONLY=1` makes the launcher use the preserved original CPU PyTorch package. The existing Flask dashboard continues to use its original checkpoint by default. To evaluate or reconstruct using a separate GPU checkpoint without replacing CPU artifacts:

```powershell
& '.\.runtime\python\python.exe' run.py evaluate.py --checkpoint runs/gpu-smoke/models/chronogan_best.pt --baseline-epochs 2 --output-dir runs/gpu-smoke
& '.\.runtime\python\python.exe' run.py verify_gpu_inference.py
```

Full training preparation, after GPU verification and smoke success:

```powershell
& '.\.runtime\python\python.exe' run.py download_full_dataset.py
& '.\.runtime\python\python.exe' run.py prepare_full_training.py
# Long training is a separate, deliberate step after reviewing resource estimates:
& '.\.runtime\python\python.exe' run.py train.py --config config.gpu-full.yaml --device cuda
```

The complete-cohort downloader uses the official public PhysioNet S3 bucket. Existing valid patient files are reused; all 40,336 must match the listed byte sizes and single-part ETag checksums. Preparation checks clinical schema, labels and hourly order, freezes disjoint patient splits, fits training-only normalization, and estimates memory/time from actual smoke observations. Full training refuses a partial cohort. `config.gpu-amp.yaml` completed a separate two-epoch BF16 trial on exactly the same patients and normalization as the FP32 smoke run, with finite losses and closely matching validation error. BF16 is enabled in `config.gpu-full.yaml`; losses remain FP32, and optional FP16 uses gradient scaling. The tiny BF16 trial did not establish a speed advantage; it verified numerical operation. Training is never launched by the preparation script.

## Real dataset and download availability

Source: [PhysioNet CinC Challenge 2019 v1.0.0](https://physionet.org/content/challenge-2019/1.0.0/). Cite Reyna et al., DOI [10.13026/v64v-d857](https://doi.org/10.13026/v64v-d857), and observe the source license.

The brief's `training_setA.zip` and `training_setB.zip` URLs are attempted automatically. At implementation time those endpoints returned HTTP 404. The downloader falls back to the working official `training/training_setA/` and `training/training_setB/` directories, validates their expected patient counts, and retrieves PSV files with bounded concurrency and retries. It does not generate substitute patient records. Existing legitimate archives in `data/` are reused after ZIP CRC validation. The full dataset must contain 20,336 set-A and 20,000 set-B records. Only patient PSV files are extracted, and `data/summary.json` records actual counts and missingness. For a smaller real-data smoke run use `python run.py download_dataset.py --limit-per-set 100`; its summary explicitly marks the subset. Download logs are in `results/download.log` for this session.

## Full reproducible experiment

```powershell
python run.py download_dataset.py
python run.py preprocess.py
python run.py train.py --config config.yaml
python run.py evaluate.py --baseline-epochs 50
python run.py predict.py data/training_setA/p000001.psv --samples 20
python run.py train.py --resume checkpoints/last.pt --epochs 100
```

Choose an existing test patient path from the checkpoint instead of the illustrative filename above. `preprocess.py` writes statistics for inspection; training independently prepares its configured cohort and embeds its exact splits and statistics in each checkpoint. Quick training uses a seeded random sample of up to 200 downloaded real patients and two epochs. The download smoke subset contains the first 100 sorted patient files from each hospital and is not representative. This is a functionality smoke test, not evidence of clinical performance. Full configuration uses all available patients and 50 epochs: finish the default downloader first to obtain the complete cohort. Resume uses the saved cohort, optimizer states and configuration, with an optional total epoch override. `checkpoints/last.pt` is the latest epoch; `models/chronogan_best.pt` is selected by validation MAE.

## Methodology

1. Patient files are split 70/15/15 before normalization or window construction. Training-only means and standard deviations are fit using finite observations. All original NaNs remain in source data. Completely absent training features get mean zero/scale one and are listed as unsupported evidence in inference metadata.
2. Non-overlapping windows initially contain up to 24 hourly rows. Padding has a separate valid-time mask. Labels never enter the imputation tensors. Windows reset recurrent state and observation elapsed time; evaluation uses the same window protocol for every method. Full-patient inference is causal, but sequences longer than the training window can differ in distribution and require separate validation.
3. A GRU-D-inspired encoder learns feature reference values and input/hidden decay from observation masks and elapsed time. A forward GRU combines hidden states and fresh Gaussian noise; its output is composed with visible observations using `M*X+(1-M)*generated`. This is a custom architecture, not an exact reproduction of GRU-D.
4. The temporal convolution discriminator predicts cell observation membership with GAIN-style partial hints. The mask is only partially revealed; unrevealed cells contribute to discrimination, and generated cells contribute to generator adversarial loss. Naturally missing sequences are never used as fully observed targets. Denoising loss targets only observed cells artificially hidden from the generator. Delta is recomputed after artificial hiding to prevent mask leakage.
5. Auxiliary losses provide low-weight temporal total variation on generated transitions and broad soft physiological bounds. Recorded observations are never clipped. The loss weights are configurable; these priors can still bias extreme values and require ablation and clinical review. Feature references are learned scalars, not external medical knowledge embeddings.
6. Validation uses fixed hidden cells and fixed noise draws. Alternating Adam updates, gradient clipping, early stopping, CPU/CUDA selection, best/latest checkpoints and training plots are implemented. Seeded runs are reproducible on the same stack subject to native CUDA nondeterminism.
7. Held-out experiments use identical real observations and patient splits for mean, forward fill, retrospective linear interpolation, a trained causal denoising GRU and ChronoGAN. The recurrent baseline is explicitly not BRITS. Random and contiguous block masking use rates 10/20/40/60%. Aggregate MAE/RMSE are standardized; per-feature errors are in original clinical units, with scored cell counts. Native missing cells never become ground truth.
8. Separate GRU sepsis models are trained for mean, forward fill, recurrent and ChronoGAN reconstructions, plus a direct missingness-aware encoder baseline. Retrospective interpolation is excluded. Reconstruction uses only historical and current values. Challenge SepsisLabel is already shifted six hours before onset; this implementation predicts that provided label at each hour, without shifting it again. Validation BCE selects downstream weights. AUROC, AUPRC, sensitivity/specificity at 0.5, Brier score and calibration bins are calculated only on test labels. Single-class discrimination metrics are reported as unavailable.
9. There is no superiority claim. A small run, pooled hourly metrics, single split/seed, and MC noise intervals do not establish generalization or clinical utility. Publication-quality conclusions need multiple seeds, patient-bootstrap intervals, external validation, missing-not-at-random analyses, calibrated uncertainty and a full BRITS comparator if desired. The application is a local research tool, not a deployed clinical service.

## Outputs and dashboard

`results/training.json`, `training.png`: actual training curves. `results/evaluation.json`, `comparison.png`, `robustness.png`: actual masking/baseline experiments. `results/sepsis.json`, `calibration.png`: actual downstream metrics/calibration. `predict.py` writes reconstructed, mask, uncertainty CSVs plus inference metadata. `smoke_test.py` automatically chooses a real held-out patient from the trained checkpoint, exports these CSVs and generates `results/reconstruction.png`. The dashboard displays only files actually generated, with blank/unavailable states otherwise; experiment results from a different checkpoint are hidden. Plotly graphs and fonts load via public CDNs; CSV/PDF functionality works without them. Graphs can be downloaded as PNG. PDF reports include inference provenance, count of imputed values, available held-out reconstruction metrics, and the research disclaimer. Metadata includes the checkpoint SHA-256, model version, processing time and sampling protocol. `requirements-lock.txt` records the installed versions from this verification run.

## Verification

`python run.py test.py` checks parsing, masks/deltas, patient disjointness, train-only statistics, padding, tensor shapes, observed-value preservation, deterministic causal-prefix behavior on CPU/CUDA, checkpoint loading, unseen patient inference, CSV serialization, baseline causality, API validation, dashboard assets, dataset checksum validation and full-cohort guards. Fixtures test mechanics only and never appear as clinical experiment results. Original CPU verification remains preserved in `results/verification.md`; current GPU and full-cohort verification is recorded in `results/gpu/final-report.md` and its companion JSON. The JUnit test report is in `results/gpu/tests.xml`.
