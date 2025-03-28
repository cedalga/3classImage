import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
import torch.nn as nn
from torchvision.models import resnet50

NUM_CLASSES = 3

class CustomResNet50(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super().__init__()
        self.base_model = resnet50(pretrained=True)
        self.base_model.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(self.base_model.fc.in_features, NUM_CLASSES)
        )
        
    def forward(self, x):
        return self.base_model(x)

# Charger le modèle
model = CustomResNet50() 
model.load_state_dict(torch.load('best_model.pth'))
model.eval() 

# Charger les noms des classes
with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

st.title('Prédiction d\'images sportives')

uploaded_file = st.file_uploader("Télécharger une image", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Image téléchargée', use_column_width=True)

if st.button('Prédire'):
    if uploaded_file is not None:
        # Prétraitement de l'image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)

        # Prédiction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()

        # Affichage des résultats
        st.write(f"Classe prédite : {class_names[predicted_class]}")
        st.write(f"Probabilités : {probabilities}") 
    else:
        st.write("Veuillez télécharger une image.")