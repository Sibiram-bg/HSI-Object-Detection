import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision  # Added for loading pre-trained models
from model import SSD300, MultiBoxLoss
from datasets_hyper import PascalVOCDataset
from utils import *
import argparse

print('Starting training with TRANSFER LEARNING strategy.')

# --- Argument Parser for Batch Size ---
parser = argparse.ArgumentParser(description='SSD300 Transfer Learning Training')
parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size for training. Default is 8.')
args = parser.parse_args()

# --- Parameters ---
data_folder = './'
keep_difficult = True
n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint =  './checkpoint_transfer.pth.tar' # Start fresh for transfer learning
batch_size = args.batch_size
start_epoch = 0
epochs = 400
epochs_since_improvement = 0
best_loss = 100.
workers = 0 # Set to 0 for Windows stability
print_freq = 5 # Print less frequently to keep the log clean
lr = 1e-3
momentum = 0.9
weight_decay = 5e-3
grad_clip = None

cudnn.benchmark = True

def main():
    global epochs_since_improvement, start_epoch, best_loss, epoch, checkpoint

    if checkpoint is None:
        print("Initializing model with PRE-TRAINED VGG-16 weights...")
        model = SSD300(n_classes=n_classes)
        
        # --- NEW: Load pre-trained backbone ---
        pretrained_vgg = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        pretrained_state_dict = pretrained_vgg.features.state_dict()
        model_state_dict = model.base.state_dict()

        for key in pretrained_state_dict:
            if key == '0.weight': continue
            if key in model_state_dict:
                model_state_dict[key] = pretrained_state_dict[key]
        
        avg_pretrained_weights = pretrained_state_dict['0.weight'].mean(dim=1, keepdim=True)
        expanded_weights = avg_pretrained_weights.repeat(1, 96, 1, 1)
        model_state_dict['stage1.0.weight'] = expanded_weights # Target the correct layer name

        model.base.load_state_dict(model_state_dict, strict=False)
        
        biases = [p for name, p in model.named_parameters() if 'bias' in name and p.requires_grad]
        not_biases = [p for name, p in model.named_parameters() if 'bias' not in name and p.requires_grad]
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print(f'\nLoaded checkpoint from epoch {start_epoch}. Best loss so far is {best_loss:.3f}.\n')
        model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']

    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    model = torch.nn.DataParallel(model)

    train_dataset = PascalVOCDataset(data_folder, split='train', keep_difficult=keep_difficult)
    val_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers, pin_memory=True)

    for epoch in range(start_epoch, epochs):
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch)
        val_loss = validate(val_loader=val_loader, model=model, criterion=criterion)
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}")
        else:
            epochs_since_improvement = 0

        # Use new checkpoint names
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best, "checkpoint_transfer.pth.tar")
        
        epochs_remaining = epochs - 1 - epoch
        print("----------------------------------------------------")
        print(f"Epoch {epoch + 1} of {epochs} complete. Best Loss: {best_loss:.3f}. ({epochs_remaining} epochs remaining)")
        print("----------------------------------------------------\n")

# You'll also need to update the save_checkpoint function to accept a filename
def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best, filename):
    state = {'epoch': epoch, 'epochs_since_improvement': epochs_since_improvement, 'loss': loss,
             'best_loss': best_loss, 'model': model, 'optimizer': optimizer}
    torch.save(state, filename)
    if is_best:
        torch.save(state, 'BEST_' + filename)

# The train and validate functions remain the same as in the corrected train.py
# (I am omitting them here for brevity but you should copy them from your working train.py)
def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)
        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        predicted_locs, predicted_scores = model(images)
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
    del predicted_locs, predicted_scores, images, boxes, labels

def validate(val_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()
    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(val_loader):
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            predicted_locs, predicted_scores = model(images)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)
            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            if i % print_freq == 0:
                print(f'[{i}/{len(val_loader)}]\t'
                      f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
    print(f'\n * VALIDATION LOSS - {losses.avg:.3f}\n')
    return losses.avg

if __name__ == '__main__':
    main()