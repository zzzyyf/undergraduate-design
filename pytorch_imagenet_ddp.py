import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from torch.nn.parallel import DistributedDataParallel as DDP

def inference(model, dataloader, device, enabled):
    sum_input = 0
    sum_right = 0
    for i, (images, targets) in enumerate(dataloader):
        # print(f'batch No.{i+1} start!')
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=enabled):
                images = images.to(device)
                # print(torch.cuda.memory_summary())
                outputs = model(images)
            
                # get accuracy
                #'''
                predictions = torch.max(outputs, 1)[1]
                batch_size = outputs.size(0)
                for j in range(batch_size):
                    if predictions[j]==targets[j]:
                        sum_right += 1
                sum_input += batch_size
                #'''
                # if i==0:
                #     break
    print(f'evaluated {sum_input} samples, top-1 accuracy: {sum_right * 1.0 / sum_input}')

def inf_proc(rank, world_size):
    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f'{rank} dist init completed.')
    datapath = '/home/zyf/Documents/ImageNet'
    transform = transforms.Compose(
            [
            # scale and normalize to inception_v3 format
            transforms.Resize((299, 299)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
            ]) 
    print(f'{rank} started.')
    dataset = datasets.ImageNet(root=datapath, split="val", transform=transform)
    print(f'{rank} dataset load complete.')
    BATCH_SIZE = int(1024 / world_size)
    sampler = torch.utils.data.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=True)
    
    #print(f'{rank} dataset ready.')
    device = rank
    model = models.inception_v3(pretrained=True, progress=False)
    model.eval()
    torch.cuda.set_device(device)
    model = model.cuda(device)

    ddp_model = DDP(model, device_ids=[rank])
    #print(f'{rank} model ready.')
    # forward pass
    inference(ddp_model, dataloader, device, False)
    #print(rank)
    inference(ddp_model, dataloader, device, True)
    dist.destroy_process_group()

def main():
    world_size = 4
    os.environ['MASTER_ADDR'] = '127.0.0.1' 
    os.environ['MASTER_PORT'] = '12345'
    mp.spawn(inf_proc, args=(world_size,), nprocs=world_size, join=True)

if __name__=='__main__':
    main()
