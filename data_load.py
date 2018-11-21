import numpy as np 
import torch
from torchvision import transforms, datasets

data_transforms = transforms.Compose([
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
						 std=[0.229, 0.224, 0.225])
	])

data_set = datasets.ImageFolder(root='./images',
								transform=data_transforms)
dataset_loader = torch.utils.data.DataLoader(data_set,
											 batch_size=4, shuffle=True,
											 num_workers=4)

print(data_set[1][0].size())