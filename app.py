import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

st.title("🧠 Diagnostic des Tumeurs Cérébrales (PFE)")
st.write("Uploadez une image IRM pour obtenir une prédiction en temps réel.")

# Charger le modèle directement depuis le dossier poussé sur GitHub
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = models.densenet121(weights=None)
    num_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 4)
    )
    
    # Le chemin exact qu'on a trouvé dans ton terminal
    MODEL_PATH = "best_brain_tumor_densenet121_pytorch.pth"
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        st.error(f"Le dossier du modèle '{MODEL_PATH}' est introuvable. Veuillez vérifier vos fichiers sur GitHub.")
        
    model.eval()
    return model

model = load_model()
classes = ["glioma", "meningioma", "notumor", "pituitary"]

# Prétraitement de l'image (mismo que l'entraînement)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choisissez une image IRM...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image IRM Chargée", use_container_width=True)
    
    if st.button("Lancer la détection 🚀"):
        with st.spinner("Analyse en cours..."):
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
            
            prediction = classes[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            st.success(f"### Résultat : {prediction.upper()}")
            st.metric(label="Indice de confiance", value=f"{confidence_score:.2f}%")
            
            # Affichage du graphique des probabilités
            all_probs = {classes[i]: float(probabilities[i].item() * 100) for i in range(len(classes))}
            st.bar_chart(all_probs)