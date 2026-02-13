import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
import sys

# Class labels

CLASS_NAMES = [‘akiec’, ‘bcc’, ‘bkl’, ‘df’, ‘mel’, ‘nv’, ‘vasc’]

CLASS_DESCRIPTIONS = {
‘akiec’: ‘Actinic Keratoses (precancerous)’,
‘bcc’: ‘Basal Cell Carcinoma’,
‘bkl’: ‘Benign Keratosis’,
‘df’: ‘Dermatofibroma’,
‘mel’: ‘Melanoma (malignant)’,
‘nv’: ‘Melanocytic Nevi (mole)’,
‘vasc’: ‘Vascular Lesion’
}

def load_model(model_path=‘ham10000_mobilenetv2.pth’):
“”“Load trained model”””
device = torch.device(“cuda” if torch.cuda.is_available() else “cpu”)

```
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

return model, device
```

def predict_image(image_path, model, device):
“”“Predict skin lesion class from image”””

```
# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = F.softmax(outputs, dim=1)[0]
    confidence, predicted_idx = torch.max(probabilities, 0)

predicted_class = CLASS_NAMES[predicted_idx.item()]
confidence_score = confidence.item()

return predicted_class, confidence_score, probabilities
```

def print_results(image_path, predicted_class, confidence, probabilities):
“”“Print prediction results”””
print(”\n” + “=”*60)
print(f”Image: {image_path}”)
print(”=”*60)
print(f”\nPrediction: {predicted_class.upper()}”)
print(f”Description: {CLASS_DESCRIPTIONS[predicted_class]}”)
print(f”Confidence: {confidence*100:.2f}%”)

```
print("\nAll class probabilities:")
for i, class_name in enumerate(CLASS_NAMES):
    prob = probabilities[i].item() * 100
    bar = "█" * int(prob / 2)
    print(f"{class_name:6s} {prob:5.2f}% {bar}")

print("\n" + "="*60)

# Medical warning
if predicted_class == 'mel':
    print("⚠️  WARNING: Melanoma detected!")
    print("   Consult a dermatologist immediately.")
elif predicted_class in ['bcc', 'akiec']:
    print("⚠️  NOTICE: Potentially concerning lesion detected.")
    print("   Recommend professional evaluation.")

print("="*60 + "\n")
```

if **name** == “**main**”:
if len(sys.argv) < 2:
print(“Usage: python predict.py <image_path> [model_path]”)
print(“Example: python predict.py lesion.jpg”)
sys.exit(1)

```
image_path = sys.argv[1]
model_path = sys.argv[2] if len(sys.argv) > 2 else 'ham10000_mobilenetv2.pth'

print("Loading model...")
model, device = load_model(model_path)

print("Analyzing image...")
predicted_class, confidence, probabilities = predict_image(image_path, model, device)

print_results(image_path, predicted_class, confidence, probabilities)
```
