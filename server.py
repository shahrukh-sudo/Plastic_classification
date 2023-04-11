from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import pickle

# Define the FastAPI app
app = FastAPI()

# Define the input and output models
class OutputData(BaseModel):
    prediction: float

# Load the pickled model
with open("C:\Users\shahr\export.pkl", "rb") as f:
    model = pickle.load(f)

# Define the API endpoint
@app.post("/predict", response_model=OutputData)
async def predict(image: UploadFile = File(...)):
    # Read the image file contents
    contents = await image.read()

    # Load the image using TensorFlow or another library of your choice
    # Here's an example using TensorFlow:
    img = tf.image.decode_jpeg(contents, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.reshape(img, [1, 224, 224, 3])
    img = tf.cast(img, tf.float32)

    # Use the model to make a prediction
    prediction = model.predict(img)[0][0]

    # Return the prediction as the API output
    return {"prediction": prediction}
