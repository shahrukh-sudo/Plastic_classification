from fastai.vision.all import *
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)

# Load the Fastai model
model_path = os.path.join("C:/", "Users", "shahr", "export.pkl")
learn = load_learner(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    
    # Save the file to disk
    filename = secure_filename('test')
    file.save(filename)
    
    # Load the image using the Fastai PILImage class
    img = PILImage.create(filename)
    
    # Make a prediction using the loaded Fastai model
    pred, label, prob = learn.predict(img)
    
    # Return the prediction as a JSON object
    response = {
        'prediction': label,
        'probability': float(prob)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)
    print('Server running at http://localhost:3000')