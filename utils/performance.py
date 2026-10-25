"""Optional, benchmark-gated execution improvements; no model/objective changes."""
from pathlib import Path
import numpy as np
import torch
from utils.cache import CachedWindows


class WorkerSafeWindows(CachedWindows):
    def __init__(self,folder):
        self.folder=str(Path(folder).resolve())
        super().__init__(self.folder)

    def __getstate__(self):
        # Windows spawn must reopen mmap files, never pickle the entire cohort.
        return {'folder':self.folder}

    def __setstate__(self,state):
        self.__init__(state['folder'])


class BatchedWindows(WorkerSafeWindows):
    def __getitems__(self,indices):
        return tuple(torch.from_numpy(np.asarray(array[indices])) for array in self.arrays)


def collate_cached_batch(batch):
    return batch


class DeferredMetrics:
    """Copy detached batch scalars together at epoch end, preserving CPU means."""
    keys=('generator','discriminator','adversarial_loss','reconstruction_mse','temporal_loss','plausibility_loss')

    def __init__(self):
        self.rows=[]

    def add(self,generator,discriminator,components):
        self.rows.append(torch.stack([generator.detach(),discriminator.detach(),*[components[key].detach() for key in self.keys[2:]]]))

    def means(self):
        # Match the original np.mean of Python float values (float64 accumulation).
        values=torch.stack(self.rows).cpu().numpy().astype(np.float64)
        return dict(zip(self.keys,values.mean(0).tolist()))
