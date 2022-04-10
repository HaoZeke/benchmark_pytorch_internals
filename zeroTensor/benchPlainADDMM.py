import torch
import itertools
import time
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare
import sys
import pickle

print('Using pytorch %s' % (torch.__version__))

shapes = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
results = []
repeats = 10
device = 'cpu'
dtype = torch.double

for device in ['cpu']:#, 'cuda']:
    for mat1_shape in shapes:
        inp_ = torch.rand(*mat1_shape, dtype=dtype, device=device, requires_grad=False)
        mat1_ = torch.randn(*mat1_shape, dtype=dtype, device=device, requires_grad=False)
        mat2_ = torch.randn(*mat1_shape, dtype=dtype, device=device, requires_grad=False)
        mezt = torch._efficientzerotensor(*mat1_shape)

        # ZT --> Efficient Zero Tensors instead of undefined tensors
        tasks = [
            ("mezt.addmm(mezt, mezt)", "ZT->inp, mat1, mat2"),
            ("inp_.addmm(mezt, mezt)", "ZT->mat1, mat2"),
            ("inp_.addmm(mat1_, mezt)", "ZT->mat2"),
            ("inp_.addmm(mat1_, mat2_)", "No ZT"),
        ]
        timers = [Timer(stmt=stmt, label=f"input dtype:{dtype} device:{device}", sub_label=f"{(mat1_shape)}",
                        description=desc, globals=globals()) for stmt, desc in tasks]

        for i, timer in enumerate(timers * repeats):
            results.append(
                timer.blocked_autorange()
            )
            print(f"\r{i + 1} / {len(timers) * repeats}", end="")
            sys.stdout.flush()

with open('with_zt_simpleaddmm.pkl', 'wb') as f:
    pickle.dump(results, f)

comparison = Compare(results)
comparison.print()
