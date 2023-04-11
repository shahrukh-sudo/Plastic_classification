
from fastai.vision.all import *
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import io

learn = load_learner(model_path)
model_path = os.path.join("C:/", "Users", "shahr", "export.pkl")


@app.route('/predict', methods=['POST'])
def predict():


    print(learn.predict('OIP.jpg'))
    response = {
        'prediction': 'oka',
        'probability': 'okkk'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=3000)
    print('Server running at http://localhost:3000')