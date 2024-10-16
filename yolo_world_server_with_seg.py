from flask import Flask, request, jsonify
import numpy as np
import time
from io import BytesIO
from PIL import Image
import sys
from ultralytics import YOLO
import torch
import clip
import argparse
from transformers import SamModel, SamProcessor
from scipy import ndimage
from torch.amp import autocast
import cv2 

args = None

class YoloWorldInference:
    def __init__(self, device="cuda"):
        # Configure and initialize the model
        self.yolo_model = YOLO("yolov8x-worldv2.pt").to(device)
        self.prev_classes = ""
        self.clip_model, _ = clip.load("ViT-B/32")
        
        self.seg_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.seg_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
        #self.seg_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        #self.seg_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.device = device
        
    @staticmethod
    def load_image(image_data):
        """Load an image from binary data."""
        pil_image = Image.open(BytesIO(image_data)).convert("RGB")
        #print("WRITING IMAGE TO: tmp.jpg")
        #cv2.imwrite("tmp.jpg",np.array(pil_image))
        return pil_image

    def infer(self, image, classes):
        """Perform inference and return bounding boxes."""
        if self.prev_classes != classes:
            self.set_classes(classes)
        print("Running YOLO model prediction...")
        with autocast('cuda'):
            results = self.yolo_model.predict(image, stream=True, max_det=args.max_det, conf=args.conf, iou=args.iou )
        print("Prediction completed.")
        torch.cuda.empty_cache()
        return results

    # Note the built in set classes method doesn't keep clip in vram
    # Here we keep the model loaded to speed up inference time
    def set_classes(self, classes):
        device = next(self.clip_model.parameters()).device
        text_token = clip.tokenize(classes).to(device)
        txt_feats = self.clip_model.encode_text(text_token).to(dtype=torch.float32)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.yolo_model.model.txt_feats = txt_feats.reshape(
            -1, len(classes), txt_feats.shape[-1]
        )
        self.yolo_model.model.names = classes
        background = " "
        if background in classes:
            classes.remove(background)
            
        yolo_model_ref = self.yolo_model.model.model
        #print("YOLO model classes: ",self.yolo_model.model.names)
        yolo_model_ref[-1].nc = len(classes)
        if self.yolo_model.predictor:
            self.yolo_model.predictor.model.names = classes
            
    def get_segmentations_from_boxes(self, image, input_boxes):
        input_boxes = input_boxes[:5]
        inputs = self.seg_processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        try: 
            image_embeddings = self.seg_model.get_image_embeddings(inputs["pixel_values"])
        except:
            image_embeddings = None 
            while image_embeddings is None: 
                image_embeddings = self.seg_model.get_image_embeddings(inputs["pixel_values"]) 
                time.sleep(0.1)
        inputs["input_boxes"].shape
        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = self.seg_model(**inputs, multimask_output=False)

        masks = self.seg_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        return masks, scores
    
    def get_projection_points(self, image, boxes):
        masks, scores = self.get_segmentations_from_boxes(image, boxes)
        points = self.calculate_center_of_mass(masks[0])
        return points
        
    
    def calculate_center_of_mass(self, masks):
        """
        Calculate the center of mass for the largest contiguous object within a binary mask.
        
        Parameters:
        - masks: A list of 2D numpy array where the mask is boolean.
        
        Returns:
        - List of list of (x, y) coordinates of the center of mass for the largest object.
        """
        center_of_masses = []
        masks = masks.cpu().detach().numpy()
        for mask in masks:
            mask = mask.squeeze()  # Remove single-dimensional entries from the shape
            labeled_mask, num_features = ndimage.label(mask)
            if num_features == 0:
                return None  # Return None if no features are found
            
            # Find the area of each feature
            area = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
            largest_feature = area.argmax() + 1  # Index of the largest feature
            
            # Calculate the center of mass for the largest feature
            center_of_mass = ndimage.center_of_mass(mask, labeled_mask, largest_feature)
            # Reverse the coordinates
            center_of_mass = (center_of_mass[1], center_of_mass[0])
            center_of_masses.append(list(center_of_mass))
        return center_of_masses

# Initialize Flask app
app = Flask(__name__)


# Initialize GLIP model, adjust paths as needed
yolo_world = YoloWorldInference()


@app.route("/process", methods=["POST"])
def process_image():
    print("Processing image...")
    start_time = time.time()

    # Receive the image
    encoded_image = request.data
    image = yolo_world.load_image(encoded_image)
    #print("Received image of size: ",image.size)

    # Get caption from request
    caption = request.args.get("caption", "")
    #print(f"Received caption: {caption}")
    caption = caption.split(",")

    boxes = []
    scores = []
    names = []

    # Process the image with Yolo World
    results = yolo_world.infer(image, caption)

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        boxes_xyxy = boxes.xyxy
        x_coordinates = boxes_xyxy[:, [0, 2]]  # x_min and x_max
        y_coordinates = boxes_xyxy[:, [1, 3]]  # y_min and y_max
        scores = boxes.conf
        names = [result.names[int(idx)] for idx in boxes.cls.tolist()]
        projection_points = []
        class_ids = result.boxes.cls 
        # Get Centriods Using SAM
        if len(boxes_xyxy) > 0:
            projection_points = yolo_world.get_projection_points(image, [boxes_xyxy.tolist()])
        for i,box in enumerate(boxes):
            class_id = int(class_ids[i].item())  # Class ID (index)
            #class_name = yolo_world.model.names[class_id]  # Class name
            class_name = yolo_world.yolo_model.model.names[class_id]
            score = scores[i].item() 
            print(f"Detected {class_name} with confidence {score:.2f}")

    # Prepare the base response data
    processing_time = time.time() - start_time
    response_data = {"caption": caption, "processing_time": processing_time}

    response_data.update(
        {
            "x_coords": x_coordinates.tolist(),
            "y_coords": y_coordinates.tolist(),
            "projection_points": projection_points,
            "scores": scores.tolist(),
            "labels": names,
            "bbox_mode": "xyxy",
        }
    )
    
    print(f"Processed image with {len(names)} objects: {', '.join(names)} in {processing_time:.2f} seconds.")
    return jsonify(response_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolo Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=5005, help="Port number")
    parser.add_argument("--max_det", type=int, default=10, help="Max detections")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.1, help="IoU threshold")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
