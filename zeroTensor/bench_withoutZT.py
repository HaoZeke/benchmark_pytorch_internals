# From anjali: https://github.com/pytorch/pytorch/pull/74444#issuecomment-1087793318
import torch
import itertools
import time
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare
import sys
import pickle
import torch.autograd.forward_ad as fwAD

print('Using pytorch %s' % (torch.__version__))

shapes = [(128, 128), (256, 256), (512, 512)]
results = []
repeats = 10
device = 'cpu'
dtype = torch.double

for device in ['cpu']:#, 'cuda']:
    for mat1_shape in shapes:
        inp_ = torch.rand(*mat1_shape, dtype=dtype, device=device, requires_grad=True)
        mat1_ = torch.randn(*mat1_shape, dtype=dtype, device=device, requires_grad=True)
        mat2_ = torch.randn(*mat1_shape, dtype=dtype, device=device, requires_grad=True)
        with fwAD.dual_level():
            inp_dual_obj = fwAD.make_dual(inp_, torch.randn_like(inp_))
            mat1_dual_obj = fwAD.make_dual(mat1_, torch.randn_like(mat1_))
            mat2_dual_obj = fwAD.make_dual(mat2_, torch.randn_like(mat2_))

        def fn(inp, mat1, mat2):
            with fwAD.dual_level():
                out=inp.addmm(mat1, mat2)

        tasks = [("fn(inp_dual_obj, mat1_dual_obj, mat2_dual_obj)", "Without ZT")]
        timers = [Timer(stmt=stmt, label=f"FWD mode AD input dtype:{dtype} device:{device}", sub_label=f"{(mat1_shape)}", description=desc, globals=globals()) for stmt, desc in tasks]

        for i, timer in enumerate(timers * repeats):
            results.append(
                timer.blocked_autorange()
            )
            print(f"\r{i + 1} / {len(timers) * repeats}", end="")
            sys.stdout.flush()

with open('without_zt.pkl', 'wb') as f:
    pickle.dump(results, f)
