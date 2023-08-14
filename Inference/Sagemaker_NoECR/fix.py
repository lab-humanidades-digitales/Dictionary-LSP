import torch

model = torch.load('.' +'/'+ 'model.pth', map_location='cpu')['model_state_dict']

torch.save(model, "out.pth")
print(model)