import json
import os
import torch
import random
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import tifffile as tiff
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
voc_labels = ('leaf_screen_hy','toyblock_screen_hy','photo_screen_hy','pen_screen_hy','leaf_real_hy','toyblock_real_hy','photo_real_hy','pen_real_hy')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}

# Color map
distinct_colors = ['#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
                   '#ffd8b1', '#e6beff','#FFFFFF']
label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}


def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = list()
    labels = list()
    difficulties = list()
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text == '1')
        label = obj.find('name').text.lower().strip()
        if label not in label_map:
            continue
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text) - 1
        ymin = int(bbox.find('ymin').text) - 1
        xmax = int(bbox.find('xmax').text) - 1
        ymax = int(bbox.find('ymax').text) - 1
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)
    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}


def create_data_lists(voc07_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)
    train_images, train_objects, test_images, test_objects = list(), list(), list(), list()

    with open(os.path.join(voc07_path, 'ImageSets/Main/trainval.txt')) as f:
        ids = f.read().splitlines()
    for id in ids:
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects['boxes']) == 0: continue
        train_objects.append(objects)
        train_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.tiff'))

    with open(os.path.join(voc07_path, 'ImageSets/Main/test.txt')) as f:
        ids = f.read().splitlines()
    for id in ids:
        objects = parse_annotation(os.path.join(voc07_path, 'Annotations', id + '.xml'))
        if len(objects['boxes']) == 0: continue
        test_objects.append(objects)
        test_images.append(os.path.join(voc07_path, 'JPEGImages', id + '.tiff'))

    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j: json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j: json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j: json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j: json.dump(test_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j: json.dump(label_map, j)
    print("Data lists created successfully.")


def calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties):
    n_classes = len(label_map)
    true_images = [i for i, labels in enumerate(true_labels) for _ in labels]
    true_images = torch.LongTensor(true_images).to(device)
    true_boxes = torch.cat(true_boxes, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    true_difficulties = torch.cat(true_difficulties, dim=0)
    det_images = [i for i, labels in enumerate(det_labels) for _ in labels]
    det_images = torch.LongTensor(det_images).to(device)
    det_boxes = torch.cat(det_boxes, dim=0)
    det_labels = torch.cat(det_labels, dim=0)
    det_scores = torch.cat(det_scores, dim=0)

    average_precisions = torch.zeros((n_classes - 1), dtype=torch.float)
    for c in range(1, n_classes):
        true_class_images = true_images[true_labels == c]
        true_class_boxes = true_boxes[true_labels == c]
        true_class_difficulties = true_difficulties[true_labels == c]
        n_easy_class_objects = (1 - true_class_difficulties).sum().item()
        true_class_boxes_detected = torch.zeros(true_class_difficulties.size(0), dtype=torch.bool).to(device)
        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0: continue

        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]

        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)

        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)
            this_image = det_class_images[d]
            object_boxes = true_class_boxes[true_class_images == this_image]
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue
            
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)
            
            original_ind = torch.LongTensor(range(true_class_boxes.size(0))).to(device)[true_class_images == this_image][ind]

            if max_overlap.item() > 0.5:
                object_difficulties_in_img = true_class_difficulties[true_class_images == this_image]
                if not object_difficulties_in_img[ind]:
                    if not true_class_boxes_detected[original_ind]:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1
                    else:
                        false_positives[d] = 1
            else:
                false_positives[d] = 1
        
        cumul_true_positives = torch.cumsum(true_positives, dim=0)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)
        cumul_recall = cumul_true_positives / n_easy_class_objects if n_easy_class_objects > 0 else torch.zeros_like(cumul_true_positives)

        recall_thresholds = torch.arange(0, 1.1, 0.1).tolist()
        precisions = torch.zeros(len(recall_thresholds), dtype=torch.float).to(device)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
        average_precisions[c - 1] = precisions.mean()

    mean_average_precision = average_precisions.mean().item()
    average_precisions = {rev_label_map[c + 1]: v.item() for c, v in enumerate(average_precisions)}
    return average_precisions, mean_average_precision

def xy_to_cxcy(xy): return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2, xy[:, 2:] - xy[:, :2]], 1)
def cxcy_to_xy(cxcy): return torch.cat([cxcy[:, :2] - (cxcy[:, 2:] / 2), cxcy[:, :2] + (cxcy[:, 2:] / 2)], 1)
def cxcy_to_gcxgcy(cxcy, priors_cxcy): return torch.cat([(cxcy[:, :2] - priors_cxcy[:, :2]) / (priors_cxcy[:, 2:] / 10), torch.log(cxcy[:, 2:] / priors_cxcy[:, 2:]) * 5], 1)
def gcxgcy_to_cxcy(gcxgcy, priors_cxcy): return torch.cat([gcxgcy[:, :2] * priors_cxcy[:, 2:] / 10 + priors_cxcy[:, :2], torch.exp(gcxgcy[:, 2:] / 5) * priors_cxcy[:, 2:]], 1)
def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]
def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection
    return intersection / union

# --- NEW, CORRECTED TRANSFORM FUNCTION ---
def transform(image, boxes, labels, difficulties, split):
    assert split in {'TRAIN', 'TEST'}
    mean = [0.485, 0.456, 0.406] * 32
    std = [0.229, 0.224, 0.225] * 32

    # Convert numpy image to tensor
    new_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
    new_boxes, new_labels, new_difficulties = boxes, labels, difficulties

    if split == 'TRAIN':
        # Random crop
        new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, new_boxes, new_labels, new_difficulties)
        
        # Random horizontal flip
        if random.random() < 0.5:
            new_image = FT.hflip(new_image)
            # Flip boxes
            old_w = new_image.size(2)
            new_boxes_clone = new_boxes.clone()
            new_boxes[:, 0] = old_w - new_boxes_clone[:, 2]
            new_boxes[:, 2] = old_w - new_boxes_clone[:, 0]

    # Resize image and boxes
    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))
    # Normalize
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties

def random_crop(image, boxes, labels, difficulties):
    original_h, original_w = image.size(1), image.size(2)
    while True:
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])
        if min_overlap is None: return image, boxes, labels, difficulties
        for _ in range(50):
            scale_h, scale_w = random.uniform(0.3, 1), random.uniform(0.3, 1)
            if not 0.5 < (scale_h / scale_w) < 2: continue
            new_h, new_w = int(scale_h * original_h), int(scale_w * original_w)
            left, top = random.randint(0, original_w - new_w), random.randint(0, original_h - new_h)
            crop = torch.FloatTensor([left, top, left + new_w, top + new_h])
            overlap = find_jaccard_overlap(crop.unsqueeze(0), boxes).squeeze(0)
            if overlap.max().item() < min_overlap: continue
            new_image = image[:, top:top + new_h, left:left + new_w]
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < left + new_w) * (bb_centers[:, 1] > top) * (bb_centers[:, 1] < top + new_h)
            if not centers_in_crop.any(): continue
            new_boxes, new_labels, new_difficulties = boxes[centers_in_crop, :], labels[centers_in_crop], difficulties[centers_in_crop]
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2]) - crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:]) - crop[:2]
            return new_image, new_boxes, new_labels, new_difficulties

def resize(image, boxes, dims=(300, 300)):
    old_h, old_w = image.size(1), image.size(2)
    new_image = F.interpolate(image.unsqueeze(0), size=dims).squeeze(0)
    old_dims = torch.FloatTensor([old_w, old_h, old_w, old_h]).unsqueeze(0)
    new_boxes = boxes / old_dims
    return new_image, new_boxes

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best):
    state = {'epoch': epoch, 'epochs_since_improvement': epochs_since_improvement, 'loss': loss, 'best_loss': best_loss, 'model': model, 'optimizer': optimizer}
    torch.save(state, 'checkpoint_ssd300.pth.tar')
    if is_best: torch.save(state, 'BEST_checkpoint_ssd300.pth.tar')

class AverageMeter(object):
    def __init__(self): self.reset()
    def reset(self): self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tiff_to_numpy(tiff_image_name):
    try:
        image = tiff.imread(tiff_image_name)
        if image.ndim == 3 and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            image = image.transpose((1, 2, 0))
        return image
    except Exception as e:
        print(f"Error reading {tiff_image_name}: {e}")
        return None