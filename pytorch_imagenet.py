import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
from PIL import Image

transform = transforms.Compose(
            [
            # scale and normalize to inception_v3 format
            transforms.Resize((299, 299)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
            ]) 
dataset = datasets.ImageNet(root="/media/zyf/2894B47B94B44CD6/Users/ZYF/Downloads/ImageNet", split="val", transform=transform)

BATCH_SIZE = 128

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = models.inception_v3(pretrained=True)
model = model.eval()
model = model.to(device)
#print(torch.cuda.memory_summary())

import torch.utils.benchmark as benchmark
num_threads = torch.get_num_threads()


def inference(model, dataloader, device, enabled):
    #sum_input = 0
    #sum_right = 0
    for i, (images, targets) in enumerate(dataloader):
        # print(f'batch No.{i+1} start!')
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=enabled):
                images = images.to(device)
                # print(torch.cuda.memory_summary())
                outputs = model(images)
            
                # get accuracy
                '''
                predictions = torch.max(outputs, 1)[1]
                batch_size = outputs.size(0)
                for j in range(batch_size):
                    if predictions[j]==targets[j]:
                        sum_right += 1
                sum_input += batch_size
                '''
                #if i==0:
                #    break
    #print(f'evaluated {sum_input} samples, top-1 accuracy: {sum_right * 1.0 / sum_input}')        

torch.cuda.reset_peak_memory_stats(device)

timer = benchmark.Timer(
    stmt='inference(model, dataloader, device, False)',
    setup='from __main__ import inference',
    globals={'model': model, 'dataloader': dataloader, 'device': device},
    #num_threads=num_threads,
    label='Inference Timing',
    sub_label='Original FP32 Inference'
)

print(timer.timeit(1))

print(torch.cuda.max_memory_allocated(device)/1024.0/1024)

torch.cuda.reset_peak_memory_stats(device)

timer = benchmark.Timer(
    stmt='inference(model, dataloader, device, True)',
    setup='from __main__ import inference',
    globals={'model': model, 'dataloader': dataloader, 'device': device},
    #num_threads=num_threads,
    label='Inference Timing',
    sub_label='Mixed Precision Inference'
)

print(timer.timeit(1))

print(torch.cuda.max_memory_allocated(device)/1024.0/1024)

