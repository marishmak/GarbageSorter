from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from fastapi.responses import JSONResponse
from PIL import Image
import os
import uvicorn

def garbage_class_recognition(image):
    with torch.no_grad():
        # Perform the forward pass
        outputs = model(image)
        # Get the predicted class index
        _, predicted = torch.max(outputs, 1)
    return predicted.item()


# Define the same preprocessing steps as used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

app = FastAPI()

num_classes = 10
id2label={0:'Battery', 1: 'Biological', 2:'Brown-glass', 3:'Cardboard', 4:'Green-glass', 5:'Metal', 6:'Paper', 7:'Plastic', 8:'Trash', 9:'White-glass'}


# Load the pre-trained Vision Transformer model
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

# Load the state dictionary into the model
model_path = os.path.join('models', 'ViT_B_16_weights.pth')
state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()


@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}


# Create a prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        print('Scanning image')
        image = Image.open(file.file).convert('RGB')
        image = transform(image).unsqueeze(0)

        # Generate caption
        caption = id2label[garbage_class_recognition(image)]

        return {"caption": caption}
        

    except Exception as e:
        return JSONResponse(status_code=500, content={"caption": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)