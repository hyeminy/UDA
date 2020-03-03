import torch
a = torch.tensor([[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0],[4.0,4.0,4.0]])
print(a)
print(torch.mean(a,0)) # 각 컬럼별 평균 계산
print(torch.mean(a,1)) # 각 행별 평균 계산