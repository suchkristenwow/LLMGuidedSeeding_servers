import json
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from geometry_msgs.msg import Point 
import numpy as np 
import cv2 
import base64
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt 
import torch  
from torch.cuda.amp import autocast 
import time  
from scipy.stats import norm 
import os 
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
import re 
import copy 
from datetime import datetime

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

app = Flask(__name__)

def plot_perimeter(ax,points): 
    for pt in points:
        ax.scatter(pt[0],pt[1],color='r') 

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
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

def is_pixel_in_mask(mask, pixel_coord):
    """
    Check if a pixel coordinate is included in the binary image mask.
    
    Args:
    - mask (np.ndarray): Binary image mask where 1 or 255 represents inclusion.
    - pixel_coord (tuple): Pixel coordinate in (row, col) format.
    
    Returns:
    - bool: True if the pixel is included in the mask, False otherwise.
    """
    row, col = pixel_coord
    
    # Check if the pixel is within the bounds of the mask
    if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
        # Check if the pixel is included in the mask
        return mask[row, col] > 0
    else:
        raise ValueError("Pixel coordinates are out of bounds.")

def pick_largest_k_masks(masks, k):
    """
    Pick the largest k masks based on the number of non-zero pixels (mask area).
    
    Parameters:
        masks (list of numpy arrays): List of binary masks (2D numpy arrays)
        k (int): Number of largest masks to return
    
    Returns:
        largest_k_masks (list of numpy arrays): List of the largest k masks
    """
    # Create a list to store the masks and their areas
    mask_area_list = []

    for mask in masks:
        # Calculate the area of the mask (number of non-zero pixels)
        mask_area = np.sum(mask > 0)
        mask_area_list.append((mask, mask_area))

    # Sort the masks by area in descending order
    mask_area_list = sorted(mask_area_list, key=lambda x: x[1], reverse=True)
    #print('mask_areas: ',[x[1] for x in mask_area_list]) 

    # Select the top k largest masks
    largest_k_masks = [mask for mask, _ in mask_area_list[:k]]

    return largest_k_masks 

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

def get_ground_masks(ground_mask,all_masks): 
    """
    Return ground_masks,non_ground_masks 
    That is, the masks that are in contact with the ground mask or not 
    """
    print("getting ground masks ...")
    ground_masks = []
    non_ground_masks =[]
    for obj_mask in all_masks:
        #print("type(obj_mask):",type(obj_mask))
        # Perform element-wise AND between ground_mask and obj_mask
        if isinstance(obj_mask,dict):
            obj_mask = obj_mask['segmentation'] 
        else: 
            if isinstance(obj_mask,list): 
                obj_mask = np.array(obj_mask) 
        #print("type(obj_mask):",type(obj_mask))
        #print("obj_mask.shape: ",obj_mask.shape)
        intersection = np.logical_and(ground_mask, obj_mask)
    
        # Check if any pixel in the intersection is non-zero
        if np.any(intersection): 
            if np.sum(obj_mask) != np.sum(ground_mask):
                #print("appending mask of shape: ",obj_mask.shape)
                ground_masks.append(obj_mask)
        else:
            #print("appending mask of shape: ",obj_mask.shape)
            non_ground_masks.append(obj_mask)
    
    #print("type(ground_mask):{}, type(non_ground_mask): {}".format(type(ground_mask),type(non_ground_masks)))
    return ground_masks,non_ground_masks 

def gaussian_downsampling(mask_list,max_masks):  
    print("Downsampling {} masks down to {} masks according to area".format(len(mask_list),max_masks))
    mask_areas = [np.sum(mask) for mask in mask_list]
    mean_area = np.mean(mask_areas)
    std_dev_area = np.std(mask_areas) 
    pdf_values = norm.pdf(mask_areas,mean_area,std_dev_area) 
    sampling_prob = pdf_values/np.sum(pdf_values) 
    sampled_mask_idx = np.random.choice(np.arange(len(mask_list)),size=max_masks,p=sampling_prob)
    sampled_masks = [mask_list[i] for i in sampled_mask_idx] 
    return sampled_masks 


class SegmentAnythingServer:
    def __init__(self): 
        torch.cuda.empty_cache() 
        sam_checkpoint = "/media/kristen/easystore2/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        device = "cuda"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.seg_anything_predictor = SamPredictor(sam)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        self.current_masks = {} 

        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat-Int4",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

        print("Done initting the model!")
        '''
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL-Chat-Int4",  # Example for smaller model
            device_map="auto",
            trust_remote_code=True,
            offload_folder="/media/kristen/easystore2/tmp/",
            torch_dtype=torch.float16  # Use reduced precision
        ).eval()
        '''

    def get_possible_object_masks(self,encoded_img,obj_name,obj_description,multi):
        print("getting possible object masks ...")
        bounding_boxes = []
        img = decode_base64_image(encoded_img)
        if not os.path.exists("/home/kristen/possible_object_mask_imgs"):
            os.mkdir("/home/kristen/possible_object_mask_imgs") 

        # Get current date and time
        current_datetime = datetime.now()
        # Print the current date and time in a specific format
        formatted_datetime = current_datetime.strftime("%H-%M-%S")
        print("writing ","/home/kristen/possible_object_mask_imgs/init_img_"+formatted_datetime+".jpg")
        cv2.imwrite("/home/kristen/possible_object_mask_imgs/init_img_"+formatted_datetime+".jpg",img) 


        query = self.tokenizer.from_list_format([
            {'image': "/home/kristen/possible_object_mask_imgs/init_img_"+formatted_datetime+".jpg"}, # Either a local path or an url
            {'text': 'Whats in this image?'},
        ])

        _, history = self.model.chat(self.tokenizer, query=query, history=None) 
        question = "The user has described a " + obj_name + " like this: " + obj_description + "\n" + "If there is a " + obj_name + " in the image, put a bounding box around it. There is at least one instance of " + obj_name  + " in this image"
        print("Prompt: ",question)
        response, _ = self.model.chat(self.tokenizer, question, history=history)  
        print("response:",response)
        bbox_pattern = re.compile(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>")         
        match = bbox_pattern.search(response) 


        if match is not None: 
            x1, y1, x2, y2 = map(int, match.groups())
            bounding_boxes.append([x1,y1,x2,y2]) 
            print("match: ",match) 
            image = self.tokenizer.draw_bbox_on_latest_picture(response, history) 
            if image:
                image.save("/home/kristen/possible_object_mask_imgs/tokenizer_annotated_img_"+formatted_datetime+".jpg") 

        if multi:
            question = "The user has described a " + obj_name + " like this: " + obj_description + "\n" + "If there is a " + obj_name + " in the image, put a bounding box around it. Ignore the regions masked by black boxes." 
            c = 0 
            while match is not None: 
                img_copy = copy.deepcopy(img) 
                for bbox in bounding_boxes:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)  # -1 thickness fills the rectangle

                print("writing: ","/home/kristen/possible_object_mask_imgs/blocked_img"+str(c)+"_" +formatted_datetime+".jpg") 
                cv2.imwrite("/home/kristen/possible_object_mask_imgs/blocked_img"+str(c)+"_" +formatted_datetime+".jpg",img_copy)
                            
                c += 1 

                print("Prompt: ",question) 

                query = self.tokenizer.from_list_format([
                    {'image': "/home/kristen/possible_object_mask_imgs/blocked_img"+str(c)+"_" +formatted_datetime+"_.jpg"}, # Either a local path or an url
                    {'text': 'Whats in this image?'},
                ])
                _, history = self.model.chat(self.tokenizer, query=query, history=None)  
                response, _ = self.model.chat(self.tokenizer, question, history=history) 
                print("response: ",response) 
                bbox_pattern = re.compile(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>")         
                match = bbox_pattern.search(response)  
                print("match: ",match)
                if match is not None: 
                    x1, y1, x2, y2 = map(int, match.groups()) 
                    bounding_boxes.append([x1,y1,x2,y2]) 

        #for each of the bounding boxes, get the corresponding image mask 
        possible_masks = []
        for bbox in bounding_boxes:
            mean_x = np.mean([x1,x2]); mean_y = np.mean([y1,y2]) 
            min_x = min([x1,x2]); max_x = max([x1,x2]); y_min = min([y1,y2]); y_max = max([y1,y2]) 
            obj_mask,_ = self.get_contour_pts(encoded_img,(mean_x,mean_y),(min_x,y_min,max_x,y_max)) 
            possible_masks.append(obj_mask) 

        return possible_masks 

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

        '''
        plt.imshow(image) 
        plot_perimeter(plt,perimeter_points) 
        plt.savefig("/home/kristen/perimeter_points.png")
        plt.close() 
        '''

        # Convert the mask to a list and ensure the values are native Python types
        obj_mask_list = best_mask.astype(int).tolist()  # Ensure values are native Python types
        return obj_mask_list, perimeter_points

    """
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

        #torch.cuda.empty_cache()
        print("writing ... /home/kristen/masked_img.jpg")
        fig,ax = plt.subplots()
        plt.imshow(image)
        show_anns(masks,ax)
        plt.axis('off')
        plt.savefig("/home/kristen/masked_img.jpg") 
        plt.close(fig) 
        

        print("returning masks... this is t: {}".format(time.time()))
        return masks 
    """

server = SegmentAnythingServer()

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the SegmentAnything server."""
    global server
    server = SegmentAnythingServer()
    print("Done Initting the server!")
    return jsonify({"status": "Class initialized"})

@app.route('/get_contour_pts', methods=['POST'])
def get_contour_pts():
    """Get contour points from the image."""
    data = request.get_json()
    encoded_image = data.get('image')
    input_point = data.get('detection')
    boundingBox = data.get('bounding_box')
    if boundingBox is None:
        raise OSError 
    obj_mask, contour_pts = server.get_contour_pts(encoded_image, input_point, boundingBox)

    return jsonify({"status": "Done", "mask": obj_mask, "contour_pts": contour_pts})

@app.route('/get_possible_object_masks', methods=['POST'])
def get_possible_object_masks():
    """Generate masks for an image."""
    data = request.get_json()
    img = data.get('image')
    object_name = data.get('object_name') 
    object_description = data.get('object_description') 
    multi = data.get('multi')
    print("received message from client!")
    #frame,colors,largest_ground_mask,grounded
    masks = server.get_possible_object_masks(img,object_name,object_description,multi)     
    tmp = []
    for mask in masks:
        if isinstance(mask,list):
            #print("len(mask): ",len(mask))
            tmp.append(mask) 
        else:
            if isinstance(mask,dict):
                #print("mask size: ",len(mask['segmentation'].tolist()))
                tmp.append(mask['segmentation'].tolist()) 
            else:
                tmp.append(mask.tolist()) 

    print("returning to client ...")
    return jsonify({"status": "Done", "masks": tmp}) 

'''
@app.route('/generate_masks', methods=['POST'])
def generate_masks():
    """Generate masks for an image."""
    data = request.get_json()
    encoded_image = data.get('image')

    masks = server.generate_masks(encoded_image)
    list_masks = [x['segmentation'].tolist() for x in masks]
    return jsonify({"status": "Done", "masks": list_masks})
'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006)
