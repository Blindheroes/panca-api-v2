import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from io import BytesIO
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np
import uvicorn
import nest_asyncio


# Mapping class names
class_mapping = {
    0: "Banana Black Sigatoka Disease",
    1: "Banana Bract Mosaic Virus Disease",
    2: "Banana Healthy Leaf",
    3: "Banana Insect Pest Disease",
    4: "Banana Moko Disease",
    5: "Banana Panama Disease",
    6: "Banana Yellow Sigatoka Disease"
}

# Banana disease model class


class BananaDiseaseModel:
    def __init__(self):
        self.model_lenet = tf.keras.models.load_model('model\model_lenet.h5')
        self.model_resnet = tf.keras.models.load_model('model\model_resnet.h5')
        self.model_inception = tf.keras.models.load_model(
            'model\model_inception.h5')

    def predict_with_voting(self, image_data):
        input_size = (128, 128)
        img = preprocess_image(image_data, input_size)

        pred_lenet = self.model_lenet.predict(img)
        pred_resnet = self.model_resnet.predict(img)
        pred_inception = self.model_inception.predict(img)

        # Aggregated prediction with soft voting
        final_pred_prob = (pred_lenet + pred_resnet + pred_inception) / 3
        final_pred_class = np.argmax(final_pred_prob, axis=1)

        # Map the prediction to class names
        return [class_mapping[class_index] for class_index in final_pred_class]


disease_model = BananaDiseaseModel()

# Preprocessing image function


def preprocess_image(image_data, target_size):
    if isinstance(image_data, Image.Image):
        img = image_data.resize(target_size)
    else:
        img = Image.fromarray(image_data).resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Read image from file function


def read_file_as_image(data) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))
        image.verify()  # Verify the image is valid
        image = Image.open(BytesIO(data))  # Reopen for further processing
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Error reading image: {e}")


# FastAPI application
app = FastAPI()


@app.get("/")
async def home():
    return {"message": "WELCOME TO BANANA DISEASE API BY PANCA. GO TO api.deteksipanca.com/docs FOR API DOCUMENTATION"}


@app.get("/ping")
async def ping():
    return {"message": "API is working fine"}


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Limit file size
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    await file.seek(0)  # Reset file pointer to start
    file_size = len(await file.read())  # Read to get the size of the file
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail="File too large. Maximum allowed size is 10MB."
        )
    # Reset file pointer to start again before reading file content
    await file.seek(0)

    try:
        if not file.content_type.startswith("image/"):
            return {"error": "Invalid file type. Please upload an image."}

        file_content = await file.read()  # Read the uploaded file
        image = read_file_as_image(file_content)
        predictions = disease_model.predict_with_voting(image)

        return {"class": predictions}
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}


# Run FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=443)
