import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import *
from datasets_hyper import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True
batch_size = 12 # You can lower this if you get memory errors, but 12 should be fine for evaluation.
workers = 0 # CORRECTED: Changed from 4 to 0 to prevent freezing on Windows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = './BEST_checkpoint_ssd300.pth.tar'

# --- CORRECTED: Fixed the loading error ---
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, weights_only=False)
model = checkpoint['model'].module
model = model.to(device)
model.eval()
print(f"\nLoaded checkpoint from epoch {checkpoint['epoch'] + 1}. Best loss so far is {checkpoint['best_loss']:.3f}.\n")


# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)

            predicted_locs, predicted_scores = model(images)
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

    # Calculate mAP
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    print("\n--- Evaluation Results ---")
    print("Average Precision (AP) for each class:")
    pp.pprint(APs)
    print(f"\nMean Average Precision (mAP): {mAP:.3f}")


if __name__ == '__main__':
    evaluate(test_loader, model)