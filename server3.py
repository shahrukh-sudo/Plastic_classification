from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import os
import joblib
from fastai.vision.all import *
import pickle
from werkzeug.utils import secure_filename
import io
import base64
from pathlib import Path
import torch
# Load the trained model
model_path = os.path.join("C:/", "Users", "shahr", "SRK")
model_inf = pickle.load(open(model_path,'rb'))
# with open(model_path, 'rb') as f:
#     model = joblib.load(f)






image_path = os.path.join("C:/", "Users", "shahr", "starbucks-coffee-cup.png")
#model_inf.predict(image_path)
# Initialize the Flask application
app = Flask(__name__)

# Define the API endpoint for predicting on images
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
  


    
# ----------------------try---------------------------

    # pil_image = Image.open(file.stream).convert('RGB')
    # image_np = np.array(pil_image)
    # image_np = np.reshape(image_np, (1, 3, image_np.shape[1], image_np.shape[2]))

    # image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().div(255)
    # file_storage = file
    # file_path = Path(file_storage.filename)
    # print(file_path)
    # image_pil = Image.open(file_path)
    # image_pil = image_pil.resize((150, 150), resample=Image.BILINEAR)
    # image_tensor = torch.load(file_path)
    # image_np = np.array(image_pil)
    # image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().div(255)
    # image_tensor = image_tensor.unsqueeze(0)
    file.save( file.filename)

    


# ----------------------try---------------------------
    

    

    
   
    
   
    print(model_inf.predict(file.filename))
    

    
    # Return the prediction as a JSON object
    response = {
        'prediction': 'okay',
        'probability': 'ok'
    }
    return jsonify(response)






#     # Get the file from the POST request
#     file = request.files['file']

#     # Open the image file
#     img = Image.open(file.stream)

# #     # Convert the image to a numpy array
# #     img_array = np.array(img)

# #     # Reshape the image to the expected dimensions for the model
# #    img_array.reshape((1, -1)) img_array = 

#     # Use the trained model to make a prediction
#     prediction = model.predict(file)
#     print(prediction)

#     # Return the predicted class label
#     return jsonify({'prediction': str(prediction[0])})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)
    print('Server running at http://localhost:3000')
