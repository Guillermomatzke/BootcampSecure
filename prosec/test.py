import torch
print(torch.cuda.is_available())  # Esto debería devolver True si CUDA está habilitado
print(torch.cuda.get_device_name(0))  # Esto debería devolver el nombre de tu GPU
