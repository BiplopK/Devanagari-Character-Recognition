# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# import io
# import torch
# import torchvision.transforms as transforms
# from model import DevanagariNetwork
# app = FastAPI()
# model=DevanagariNetwork()
# model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
# model.eval()

# # Define image transformations
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# @app.post("/predict/")
# async def predict_image(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read())).convert('L')
#     image = transform(image).unsqueeze(0)  # Add batch dimension
    
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)
    
#     return {"prediction": int(predicted.item())}



from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from torchvision import transforms,datasets
from PIL import Image
import torch
import io

# Import the model from model.py
from model import DevanagariNetwork

# Load the pre-trained model
model = DevanagariNetwork()
model.load_state_dict(torch.load("model/best_model.pth", map_location=torch.device('cpu'),weights_only=True))

# Define the preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Load the dataset to retrieve class labels
dataset = datasets.ImageFolder(root='./combine_dataset', transform=transform)

# Reverse the class_to_idx mapping to create an idx_to_class mapping
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
model.eval()



# Initialize FastAPI app
app = FastAPI()

# Health check endpoint
@app.get("/ceck")
async def check():
    return JSONResponse(content={"status": "ok"})

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image
        image = transform(image).unsqueeze(0)

        # Model inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
            
        predicted_label = idx_to_class.get(predicted.item(), "Unknown")
        # Return prediction as JSON
        return JSONResponse(content={"predicted_class": predicted_label})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run the server with:
# uvicorn main:app --reload
