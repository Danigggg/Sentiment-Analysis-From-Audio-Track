import torch

tensor1 = torch.randn(3,4)
print(tensor1)

tensor2 = torch.randn(4)
print(tensor2)

result = torch.matmul(tensor1,tensor2)

print(result)