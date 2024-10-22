from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)
import re 
import cv2 
from flask import Flask, request, jsonify 
import base64
import numpy as np 
from datetime import datetime 
from PIL import Image
import torchvision.transforms as transforms
from modelscope import snapshot_download 

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),  # Adjust based on model input size
        transforms.ToTensor(),
    ])

    # Apply transformations
    image_tensor = preprocess(image)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, C, H, W]
    
    return image_tensor


def decode_base64_image(encoded_image):
    """Convert base64 string to an OpenCV image and flip the colors from BGR to RGB."""
    image_data = base64.b64decode(encoded_image)
    np_arr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image

class QwenServer: 
    def __init__(self):  
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

        print("loading model ...")
        #model_dir = snapshot_download('qwen/Qwen-VL-Chat') 
        #print("took the snapshot :)") 
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        # Check if all parameters are properly loaded
        '''
        for name, param in self.model.named_parameters():
            print(f"Parameter: {name}, device: {param.device}")

        print(f"Model's first parameter is on device: {next(self.model.parameters()).device}")  

        for name, param in self.model.named_parameters():
            print(f"{name} is on {param.device}")
        '''

    def first_query(self, image_path):
        # Pass the image path (string), as the tokenizer expects a path or URL, not a tensor
        query = self.tokenizer.from_list_format([
            {'image': image_path},  # Pass the image path instead of the tensor
            {'text': 'Describe the objects you see in the image.'}
        ]) 
        response, history = self.model.chat(self.tokenizer, query=query, history=None) 
        print("first response: ", response)  
        return response, history

    def second_query(self,question,image_path): 
        _,history = self.first_query(image_path)
        response,history  = self.model.chat(self.tokenizer,question,history=history) 
        return response, history
    
    def get_bounding_box(self, question, img):
        now = datetime.now() 
        formatted_time = now.strftime("%B%d_%Y-%H-%M-%S") 
        image_path = "/home/kristen/qwen_server_debug_imgs/" + formatted_time + "_client_img.jpg"  

        cv2.imwrite(image_path, img)
        print(f"Saved image to {image_path}")

        # Preprocess the image and move it to the same device as the model
        image_tensor = preprocess_image(image_path)
        
        # Move the image tensor to the device of the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device)

        #print(f"Image tensor is on device: {image_tensor.device}")

        torch.cuda.empty_cache()
        # First query using the image path
        print("calling first query ...")
        _, history = self.first_query(image_path)

        torch.cuda.empty_cache()  # Clear unused memory 
        
        print("calling second query!")
        response,history  = self.model.chat(self.tokenizer,question,history=history) 
        
        torch.cuda.empty_cache()  # Clear unused memory

        # Search for bounding box in the response
        bbox_pattern = re.compile(r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>")         
        match = bbox_pattern.search(response) 
        if match is None:
            print("No match!") 
            return None
        
        x1, y1, x2, y2 = map(int, match.groups()) 
        print("got match: {},{},{},{}".format(x1,y1,x2,y2))
        # Write result if bounding box is found
        self.write_result_img(response, history, formatted_time=formatted_time)
        return (x1, y1, x2, y2)  
       
    def write_result_img(self,response,history,formatted_time=None):
        image = self.tokenizer.draw_bbox_on_latest_picture(response, history)
        if image:
            if formatted_time is None: 
                print("writing out.jpg ...")
                image.save('out.jpg')
            else:
                print("writing /home/kristen/qwen_server_debug_imgs/"+formatted_time+"_annotated_img.jpg ...")
                image.save("/home/kristen/qwen_server_debug_imgs/"+formatted_time+"_annotated_img.jpg")
        else:
            print("no box") 

app = Flask(__name__)
server = QwenServer()

@app.route('/get_bounding_box', methods=['POST'])
def get_bounding_box():
    try:
        data = request.get_json() 
        encoded_img = data.get('image')
        print("received an image from the client!")
        image = decode_base64_image(encoded_img) 
        question = data.get('query')
        boundingBox = server.get_bounding_box(question, image)
        if boundingBox is None:
            return jsonify({"status": "No bounding box found", "boundingBox": None}), 404
        return jsonify({"status": "Done", "boundingBox": boundingBox}), 200
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"status": "Error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5007) 
    '''
    question = """I'm looking for a spot robot. The user has provided this description of a spot robot: The boston dynamics spot is a boxy yellow robot with four black legs. It can stand, walk, or lay down. It has a large black plastic box on it's back. There are several 
    cameras and sensors on it, including an Ouster lidar on the top. 
    Draw a bounding box around the spot robot in this image. 
    """
    image = cv2.imread("/home/kristen/ex_image.jpg")
    server.get_bounding_box(question,image)
    '''
