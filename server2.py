from fastapi import FastAPI, File, UploadFile
import joblib
import uvicorn
import numpy as np
from PIL import Image
import os
import joblib
import pickle
app = FastAPI()


# load the model
model_path = os.path.join("C:/", "Users", "shahr", "SRK")
model = pickle.load(open(model_path,'rb'))
# Define API route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # read the image and convert it to numpy array
    image = Image.open(file.file)
    image = image.resize((224, 224))
    image = np.array(image)

    # normalize the image
    image = image / 255.0

    # make prediction
    prediction = model.predict(np.array([image]))

    # return prediction
    return {"prediction": prediction[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Server running at http://localhost:8000/")
