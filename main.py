import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from model import ANFIS  

# -------------------------------
# Load PCA transformer and trained model weights
# -------------------------------
# Load the PCA transformer saved as "pca_transform.pkl"
pca_transform = joblib.load("pca_transform.pkl")

# Create the multi-output ANFIS model.
# Assume PCA reduction to 9 components.
pca_components = 9
model = ANFIS(input_dim=pca_components, num_mfs=2, output_dim=3)
model.load_state_dict(torch.load("final_model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------------------
# Define the 27 Dark Triad Questions
# -------------------------------
questions = {
    "M1": "It's not wise to tell your secrets.",
    "M2": "I like to use clever manipulation to get my way.",
    "M3": "Whatever it takes, you must get the important people on your side.",
    "M4": "Avoid direct conflict with others because they may be useful in the future.",
    "M5": "It's wise to keep track of information that you can use against people later.",
    "M6": "You should wait for the right time to get back at people.",
    "M7": "There are things you should hide from other people because they don't need to know.",
    "M8": "Make sure your plans benefit you, not others.",
    "M9": "Most people can be manipulated.",
    "N1": "People see me as a natural leader.",
    "N2": "I hate being the center of attention.",
    "N3": "Many group activities tend to be dull without me.",
    "N4": "I know that I am special because everyone keeps telling me so.",
    "N5": "I like to get acquainted with important people.",
    "N6": "I feel embarrassed if someone compliments me.",
    "N7": "I have been compared to famous people.",
    "N8": "I am an average person.",
    "N9": "I insist on getting the respect I deserve.",
    "P1": "I like to get revenge on authorities.",
    "P2": "I avoid dangerous situations.",
    "P3": "Payback needs to be quick and nasty.",
    "P4": "People often say I'm out of control.",
    "P5": "It's true that I can be mean to others.",
    "P6": "People who mess with me always regret it.",
    "P7": "I have never gotten into trouble with the law.",
    "P8": "I enjoy having sex with people I hardly know.",
    "P9": "I'll say anything to get what I want."
}

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.title("Dark Triad Trait Predictor")

st.markdown("""
This app asks you 27 questions related to personality traits (Machiavellianism, Narcissism, and Psychopathy) from the Short Dark Triad measure. 
Please answer each question on a scale of **1 (Disagree)** to **5 (Agree)**.
""")

with st.form(key="dark_triad_form"):
    responses = []
    st.header("Please answer the following questions:")
    for code, question in questions.items():
        response = st.slider(f"{code}: {question}", min_value=1, max_value=5, value=3, step=1)
        responses.append(response)
    
    submit_button = st.form_submit_button(label="Predict Dark Triad Traits")

if submit_button:
    responses = np.array(responses).reshape(1, -1)
    responses_norm = (responses - 1) / 4.0  # Normalize to [0,1]
    
    # Apply the PCA transformation (must be the same as during training)
    responses_reduced = pca_transform.transform(responses_norm)  # shape: (1, pca_components)
    
    # Convert to tensor and predict using the ANFIS model.
    input_tensor = torch.tensor(responses_reduced, dtype=torch.float32)
    with torch.no_grad():
        raw_output = model(input_tensor)
        # Apply softmax to obtain probabilities for each trait.
        probabilities = F.softmax(raw_output, dim=1).cpu().numpy()[0]
    
    # Determine interpretation based on thresholds
    # Here we use 50% as a threshold; adjust based on your domain knowledge.
    thresholds = [0.5, 0.5, 0.5]
    interpretations = []
    trait_names = ["Machiavellianism", "Narcissism", "Psychopathy"]
    for prob, thresh, name in zip(probabilities, thresholds, trait_names):
        if prob >= thresh:
            interpretations.append(f"High {name}")
        else:
            interpretations.append(f"Low {name}")
    
    # Also determine the dominant trait.
    dominant_trait = trait_names[np.argmax(probabilities)]
    
    st.subheader("Your Predicted Dark Triad Trait Probabilities")
    st.write(f"**Machiavellianism:** {probabilities[0]*100:.2f}%")
    st.write(f"**Narcissism:** {probabilities[1]*100:.2f}%")
    st.write(f"**Psychopathy:** {probabilities[2]*100:.2f}%")
    
    st.subheader("Trait Interpretations")
    for interp in interpretations:
        st.write(f"- {interp}")
    st.write(f"\n**Dominant Trait:** {dominant_trait}")
    
    st.markdown("""
    **Note:** This model was trained in an unsupervised manner with pseudo‑labels.
    With further calibration and validation, these probabilities and interpretations can be refined.
    """)
    
    st.success("Prediction complete!")