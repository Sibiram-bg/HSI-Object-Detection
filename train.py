import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make sure this is set for your GPU
import torch
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets_hyper import PascalVOCDataset
from utils import *
import argparse  # Added for command-line arguments

print('there are no enhancement of brightness and saturation,no flip')

# --- Argument Parser for Batch Size ---
parser = argparse.ArgumentParser(description='SSD300 Training')
parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size for training. Default is 8.')
args = parser.parse_args()

# Data parameters
data_folder = './'
keep_difficult = True

# Model parameters
n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint =  './BEST_checkpoint_ssd300.pth.tar'  # Set to './BEST_checkpoint_ssd300.pth.tar' to resume training
batch_size = args.batch_size  # Batch size is now set from the command line
start_epoch = 0
epochs = 400
epochs_since_improvement = 0
best_loss = 100.
workers = 4
print_freq = 1
lr = 1e-4
momentum = 0.9
weight_decay = 5e-3
grad_clip = None

cudnn.benchmark = True


def main():
    """
    Training and validation.
    """
    global epochs_since_improvement, start_epoch, label_map, best_loss, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SSD300(n_classes=n_classes)
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint, weights_only=False)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['best_loss']
        print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
        model = checkpoint['model'].module
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    model = torch.nn.DataParallel(model)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    val_dataset = PascalVOCDataset(data_folder,
                                   split='test',
                                   keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # One epoch's validation
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion)

        # Did validation loss improve?
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)
        
        # --- MODIFIED: Print summary at the end of the epoch ---
        epochs_remaining = epochs - 1 - epoch
        print("----------------------------------------------------")
        print(f"Epoch {epoch + 1} of {epochs} complete. Best Loss: {best_loss:.3f}. ({epochs_remaining} epochs remaining)")
        print("----------------------------------------------------\n")


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels


def validate(val_loader, model, criterion):
    """
    One epoch's validation.
    """
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
                print('[{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))

    print('\n * VALIDATION LOSS - {loss.avg:.3f}\n'.format(loss=losses))
    return losses.avg


if __name__ == '__main__':
    main()
