import torch
x=torch._efficientzerotensor(4, 4)
y=torch.zeros(4, 4)
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as p:
    torch.addmm(y, x, y)
print(p.key_averages().table(
    sort_by="self_cpu_time_total", row_limit=-1))
