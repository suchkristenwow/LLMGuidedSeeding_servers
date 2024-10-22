from flask import Flask, request, jsonify  
import numpy as np 
import base64 
import cv2  
from segment_anything import sam_model_registry, SamPredictor 
from torch.cuda.amp import autocast
import time 
import matplotlib.pyplot as plt 
import torch 



def show_anns(anns,ax):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    ax.imshow(img)
    
def decode_base64_image(encoded_image):
    """Convert base64 string to an OpenCV image and flip the colors from BGR to RGB."""
    image_data = base64.b64decode(encoded_image)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image


def filter_bottom_masks(masks): 
    filtered_masks = []

    for mask in masks: 
        if isinstance(mask,dict): 
            mask = mask['segmentation']
        # Get the height of the mask
        height = mask.shape[0]

        # Calculate the cutoff for the top 1/4th of the image
        top_region = height // 4

        # Check if any pixel in the top 1/4th is non-zero
        if not np.any(mask[:top_region, :]):
            # If no pixels are non-zero in the top 1/8th, keep the mask
            filtered_masks.append(mask)

    return filtered_masks

class SegmentAnythingServer:
    def __init__(self): 
        sam_checkpoint = "/media/kristen/easystore3/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.seg_anything_predictor = SamPredictor(sam)
 
    def get_contour_pts(self, encoded_image, detection_centroid, boundingBox):
        """Get contour points from an image."""
        image = decode_base64_image(encoded_image)
        self.seg_anything_predictor.set_image(image)
        
        boundingBox_height = boundingBox[1] - boundingBox[3] 
        boundingBox_width = boundingBox[2] - boundingBox[0] 

        #(detection.y, detection.x)
        ptW = np.array([detection_centroid[1],detection_centroid[0] - int(boundingBox_height/ 4)]) 
        ptN = np.array([detection_centroid[1] - int(boundingBox_width / 4),detection_centroid[0]]) 
        ptE = np.array([detection_centroid[1], detection_centroid[0] + int(boundingBox_height / 4)])
        ptS = np.array([detection_centroid[1] + int(boundingBox_width / 4), detection_centroid[0]])  

        input_point = np.array([ptW, ptN, ptE, ptS]).reshape((4, 2))  # Shape (4, 2)

        # Correct the shape of input_label to be (4, 1) instead of (1, 4)
        input_label = np.array([1, 1, 1 ,1])

        with autocast():
            masks, scores, _ = self.seg_anything_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )

        # Select the best mask  
        best_mask = masks[np.argmax(scores)]

        if np.sum(best_mask) == 0:
            max_size = 0
            for mask in masks:
                mask_size = np.sum(mask)
                if mask_size > max_size:
                    max_size = mask_size
                    best_mask = mask

        if not np.any(best_mask):
            raise OSError("No valid mask found")

        mask_uint8 = (best_mask * 255).astype(np.uint8)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, []

        largest_contour = max(contours, key=cv2.contourArea)
        perimeter_px_coords = largest_contour.reshape(-1, 2)

        # Convert NumPy array to a Python list of tuples with native int type
        perimeter_points = [(int(pt[0]), int(pt[1])) for pt in perimeter_px_coords]

        # Convert the mask to a list and ensure the values are native Python types
        obj_mask_list = best_mask.astype(int).tolist()  # Ensure values are native Python types
        return obj_mask_list, perimeter_points
     
    def generate_masks(self, encoded_image):
        #Generate masks for an image. 

        image = decode_base64_image(encoded_image)
        print("hold on ... this part usually takes a while. this is time: {}".format(time.time()))
        try: 
            t0 = time.time() 
            with autocast(): 
                masks = self.mask_generator.generate(image) 
        except torch.cuda.OutOfMemoryError:
            print("Out of memory, retrying with resized image ...")
            # Resize the image to reduce memory usage
            image = cv2.resize(image, (int(image.shape[1] // 2), int(image.shape[0] // 2)))  # Half the resolution
            with autocast():
                masks = self.mask_generator.generate(image)
        t1 = time.time() 
        print("It took {} seconds to generate the masks".format(np.round(t1 -t0)))  

        ##torch.cuda.empty_cache()
        print("writing ... /home/kristen/masked_img.jpg")
        fig,ax = plt.subplots()
        plt.imshow(image)
        show_anns(masks,ax)
        plt.axis('off')
        plt.savefig("/home/kristen/masked_img.jpg") 
        plt.close(fig) 
        
        print("returning masks... this is t: {}".format(time.time()))
        return masks 

app = Flask(__name__)
server = SegmentAnythingServer() 

@app.route('/get_contour_pts', methods=['POST'])
def get_contour_pts():
    data = request.get_json()  
    encoded_img = data.get('image')
    detection_center = data.get('detection')
    boundingBox = data.get('bounding_box') 
    obj_mask,contour_pts = server.get_contour_pts(encoded_img,detection_center,boundingBox)
    return jsonify({"status": "Done", "mask":obj_mask, "contour_pts":contour_pts})  

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=5006)

