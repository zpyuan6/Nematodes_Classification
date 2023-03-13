import os
import tqdm
import random
 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def load_data(train_dir, batch_size):
    train_and_val_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize([640,640]),
        transforms.ToTensor()
    ]))

    samples_number = len(train_and_val_dataset)
    val_index = random.sample(range(1,samples_number), 0.1*samples_number)

    f = open(os.path.join(train_dataset,"val_index.txt"),"w")
    f.write(str(val_index))
    f.close()

    val_dataset = data.Subset(train_and_val_dataset, val_index)
    train_dataset = data.Subset(train_and_val_dataset, list(set(range(1,samples_number)).difference(set(val_index))))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader,val_dataloader,train_dataset,val_dataset



def train_model(model, optimizer, train_dataload, device, epoch):
    model.train()

    sum_loss = 0

    total_num = len(train_dataload.dataset)
    step_num = len(train_dataload)
    print("training dataset total images: {}, step number: {}".format(total_num, step_num))

    with tqdm.tqdm(total = step_num) as pbar:
        pbar.set_description('Training:')
        for batch_idx, (data, target) in enumerate(train_dataload):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device == xm.xla_device():
                xm.mark_step()
            
            print_loss = loss.data.item()
            sum_loss += print_loss
            
            pbar.set_description('Training Epoch: {} Loss: {:.6f}'.format(
                    epoch, loss.item()))
            pbar.update(1)
                       
    ave_loss = sum_loss / len(train_dataload)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

    return ave_loss

def val_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        with tqdm.tqdm(total = len(test_loader)) as pbar:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                _, pred = torch.max(output.data, 1)
                correct += torch.sum(pred == target)
                print_loss = loss.data.item()
                test_loss += print_loss
                pbar.update(1)

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))

        return avgloss, correct, acc

if __name__ == "__main__":
    TRAIN_DIR = "F:\\nematoda\I-Nema\\train"
    TEST_DIR = "F:\\nematoda\I-Nema\\val"
    LEARN_RATE = 1e-4
    BATCH_SIZE = 32
    NUM_EPOCHES = 50

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader, val_dataloader,train_dataset,val_dataset = load_data(TRAIN_DIR,BATCH_SIZE)
    
    
    model = models.mobilenet_v3_large(num_classes=102)
    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(),LEARN_RATE)

    for epoch in range(NUM_EPOCHES):
        train_avgloss = train_model(model, optimizer, train_dataloader, DEVICE, epoch)
        val_avgloss, correct, acc = val_model(model, DEVICE, val_dataloader)

        if epoch % 10 == 0:
            torch.save(model, 'model_{}.pth'.format(str(epoch)))
    
    
    