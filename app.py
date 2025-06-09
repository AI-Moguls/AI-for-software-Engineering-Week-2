import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- Model Definition (from your notebook) ---
class HybridModel(nn.Module):
    def __init__(self, num_classes, meta_size):
        super().__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.cnn(img)
        meta_features = self.meta_fc(meta)
        combined = torch.cat((img_features, meta_features), dim=1)
        return self.classifier(combined)

# --- Class and Metadata Setup (from your notebook) ---
dx_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
classes = list(dx_dict.keys())
class_names = [dx_dict[k] for k in classes]
sex_categories = ['male', 'female', 'unknown']

# Use the exact localization categories from your notebook/training
localization_categories = [
    'abdomen', 'acral', 'back', 'chest', 'ear', 'face', 'foot', 'genital', 'hand',
    'lower extremity', 'neck', 'scalp', 'trunk', 'upper extremity', 'unknown'
]
meta_size = 1 + len(sex_categories) + len(localization_categories)  # 1+3+15=19

# --- Load Model ---
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = HybridModel(num_classes=len(classes), meta_size=meta_size)
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

model = load_model()

# --- Image Transform ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Streamlit UI ---
st.title("Skin Cancer Classifier")
st.write("Upload a skin lesion image and enter metadata to predict the diagnosis.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
age = st.number_input("Age", min_value=0, max_value=100, value=50)
sex = st.selectbox("Sex", sex_categories)
localization = st.selectbox("Localization", localization_categories)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Add a Run button
    if st.button("Run"):
        # Preprocess image
        img_tensor = val_transform(image).unsqueeze(0)

        # Preprocess metadata
        age_norm = age / 100.0
        sex_encoded = [1 if sex == cat else 0 for cat in sex_categories]
        loc_encoded = [1 if localization == cat else 0 for cat in localization_categories]
        meta_tensor = torch.tensor([age_norm] + sex_encoded + loc_encoded, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor, meta_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            st.subheader(f"Prediction: {pred_class}")
            st.write("Probabilities:")
            for i, cname in enumerate(class_names):
                st.write(f"{cname}: {probs[i]:.2%}")