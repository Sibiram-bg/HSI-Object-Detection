from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
import glob
from utils import tiff_to_numpy
import torchvision.transforms.functional as FT
import numpy as np
import argparse

# --- GPU and Model Loading ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = 'BEST_checkpoint_ssd300.pth.tar'
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, weights_only=False)
model = checkpoint['model'].module
model = model.to(device)
model.eval()
print(f"\nLoaded checkpoint from epoch {checkpoint['epoch'] + 1}. Best loss so far is {checkpoint['best_loss']:.3f}.\n")

# --- Transforms ---
mean=[0.485, 0.456, 0.406]*32
std=[0.229, 0.224, 0.225]*32

# --- Helper Functions ---
def resize_hyper(image, dims=(300, 300)):
    image = image.unsqueeze(0)
    new_image = F.interpolate(image,size=dims)
    return new_image.squeeze(0)

def create_false_color_image(tiff_numpy):
    red_band = tiff_numpy[:, :, 60]
    green_band = tiff_numpy[:, :, 40]
    blue_band = tiff_numpy[:, :, 20]
    false_color_img = np.stack([red_band, green_band, blue_band], axis=-1)
    false_color_img = (false_color_img / false_color_img.max() * 255).astype(np.uint8)
    return Image.fromarray(false_color_img)

# --- Main Detection Function ---
def detect(tiff_image_path, min_score, max_overlap, top_k, suppress=None):
    tiff_numpy = tiff_to_numpy(tiff_image_path)
    if tiff_numpy is None: return
    original_image = create_false_color_image(tiff_numpy)
    image = torch.from_numpy(tiff_numpy).permute(2,0,1).float()/255.
    image = resize_hyper(image, dims=(300, 300))
    image = FT.normalize(image, mean=mean, std=std)
    image = image.to(device)
    predicted_locs, predicted_scores = model(image.unsqueeze(0))
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)
    det_boxes = det_boxes[0].to('cpu')
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    if det_labels == ['background']: return original_image
    
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    for i in range(det_boxes.size(0)):
        if suppress is not None and det_labels[i] in suppress: continue
        
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]], width=2)
        
        text = f"{det_labels[i].upper()} {det_scores[0][i]:.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2]
        text_height = text_bbox[3]
        
        box_top = box_location[1]
        text_y_position = box_top - text_height - 2
        
        if text_y_position < 0:
            text_y_position = box_top + 2
            
        textbox_location = [box_location[0], text_y_position, box_location[0] + text_width + 4, text_y_position + text_height]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])

        # --- THIS LINE IS CHANGED ---
        draw.text(xy=(box_location[0] + 2, text_y_position), text=text, fill='black', font=font)
        
    del draw
    return annotated_image

# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SSD300 Detection')
    parser.add_argument('--dataset_path', required=True, help='Path to the VOC2007 dataset folder.')
    args = parser.parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')

    with open('./TEST_images.json','r') as f:
        image_paths = json.load(f)

    for i, img_path in enumerate(image_paths):
        img_filename = os.path.basename(img_path)
        local_img_path = os.path.join(args.dataset_path, 'JPEGImages', img_filename)
        
        if os.path.exists(local_img_path):
            print(f"Processing image {i+1}/{len(image_paths)}: {img_filename}")
            annotated_image = detect(local_img_path, min_score=0.2, max_overlap=0.5, top_k=200)
            if annotated_image:
                base_filename = os.path.splitext(img_filename)[0]
                annotated_image.save(f'./results/{base_filename}.png')
        else:
            print(f"Warning: Could not find image {img_filename} at the specified path: {local_img_path}")