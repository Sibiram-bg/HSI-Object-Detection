import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import *
from datasets_hyper import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

pp = PrettyPrinter()

# Parameters
data_folder = './'
keep_difficult = True
batch_size = 12
workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- MODIFIED: Points to the new transfer learning checkpoint ---
checkpoint_path = './BEST_checkpoint_transfer.pth.tar'

print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, weights_only=False)
model = checkpoint['model'].module
model = model.to(device)
model.eval()
print(f"\nLoaded checkpoint from epoch {checkpoint['epoch'] + 1}. Best loss so far is {checkpoint['best_loss']:.3f}.\n")

# ... (The rest of the script is the same as your corrected eval.py) ...
# I am including the full code for clarity.

test_dataset = PascalVOCDataset(data_folder, split='test', keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model):
    model.eval()
    det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties = [], [], [], [], [], []

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(device)
            predicted_locs, predicted_scores = model(images)
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores, min_score=0.01, max_overlap=0.45, top_k=200)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]
            det_boxes.extend(det_boxes_batch); det_labels.extend(det_labels_batch); det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes); true_labels.extend(labels); true_difficulties.extend(difficulties)

    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)
    
    print("\n--- Evaluation Results (Transfer Learning Model) ---")
    print("Average Precision (AP) for each class:")
    pp.pprint(APs)
    print(f"\nMean Average Precision (mAP): {mAP:.3f}")

if __name__ == '__main__':
    evaluate(test_loader, model)