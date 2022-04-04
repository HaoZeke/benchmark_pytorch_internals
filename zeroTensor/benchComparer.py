# From anjali: https://github.com/pytorch/pytorch/pull/74444#issuecomment-1087793318
from torch.utils.benchmark import Timer
from torch.utils.benchmark import Compare
import pickle

with open('with_zt.pkl', 'rb') as f:
    after_results = pickle.load(f)

with open('without_zt.pkl', 'rb') as f:
    before_results = pickle.load(f)

comparison = Compare(after_results + before_results)
comparison.print()
