import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import random
from model import ANFIS  

# -------------------------------
# Load the PCA transformer and trained model weights
# -------------------------------
# PCA transformer was saved as "pca_transform.pkl" during training.
pca_transform = joblib.load("pca_transform.pkl")
# Assume PCA reduction from 27 to 9 components.
pca_components = 9

# Create the ANFIS model with 3 outputs (M, N, P)
model = ANFIS(input_dim=pca_components, num_mfs=2, output_dim=3)
model.load_state_dict(torch.load("final_model_weights.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------------------
# Define Dark Triad Questions and Informational Content
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

# Additional informational content (for the "Learn More" and "Help" tabs)
learn_more = """
### About the Dark Triad Traits

- **Machiavellianism:** Characterized by manipulative, strategic, and often self-serving behavior.
- **Narcissism:** Involves grandiosity, a need for admiration, and a sense of superiority.
- **Psychopathy:** Marked by impulsivity, lack of empathy, and antisocial tendencies.

**Further Reading and Resources:**
- [Paulhus & Jones (2011) - Introducing a Short Measure of the Dark Triad](https://www.researchgate.net/publication/271258362_Introducing_a_short_measure_of_the_Dark_Triad)
- [The Dark Triad in Everyday Life](https://www.apa.org/monitor/2016/12/dark-triad)
- [Psychology Today – The Dark Triad](https://www.psychologytoday.com/us/basics/dark-triad)

These resources offer deeper insights into how these traits manifest and their implications in personal and social contexts.
"""

helpline_info = """
### If You Need Help

If you're experiencing distress or need someone to talk to, please consider reaching out to professional help or helpline services:

- **India:** Tele Manas 14416 by NIMHANS Bengaluru
- **United States:** National Suicide Prevention Lifeline: 988 or 1-800-273-8255  
- **United Kingdom:** Samaritans: 116 123  
- **Australia:** Lifeline Australia: 13 11 14  

For other regions, please refer to your local crisis intervention resources.
"""

# -------------------------------
# Streamlit App Layout with Tabs
# -------------------------------
st.title("Dark Triad Trait Predictor and Information Portal")

# Create tabs for Prediction, Learn More, and Help
tab1, tab2, tab3 = st.tabs(["Predict", "Learn More", "Help"])

with tab1:
    st.header("Predict Your Dark Triad Traits")
    st.markdown("""
    Please answer the 27 questions below. Your responses will be used to estimate your tendencies related to Machiavellianism, Narcissism, and Psychopathy.
    """)
    with st.form(key="dark_triad_form"):
        responses = []
        for code, question in questions.items():
            response = st.slider(f"{code}: {question}", min_value=1, max_value=5, value=3, step=1)
            responses.append(response)
        submit_button = st.form_submit_button(label="Predict Traits")
    
    if submit_button:
        responses = np.array(responses).reshape(1, -1)
        responses_norm = (responses - 1) / 4.0  # Normalize responses to [0,1]
        responses_reduced = pca_transform.transform(responses_norm)  # Apply PCA transformation
        input_tensor = torch.tensor(responses_reduced, dtype=torch.float32)
        with torch.no_grad():
            raw_output = model(input_tensor)
            probabilities = F.softmax(raw_output, dim=1).cpu().numpy()[0]
        
        # Interpret probabilities: here, higher than 50% is considered high.
        thresholds = [0.5, 0.5, 0.5]
        interpretations = []
        trait_names = ["Machiavellianism", "Narcissism", "Psychopathy"]
        for prob, thresh, name in zip(probabilities, thresholds, trait_names):
            if prob >= thresh:
                interpretations.append(f"High {name}")
            else:
                interpretations.append(f"Low {name}")
        
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
        **Note:** This model was trained in an unsupervised manner using pseudo‑labels.
        The provided predictions and interpretations are for informational purposes only.
        """)
        st.success("Prediction complete!")
        
with tab2:
    st.header("Learn More About the Dark Triad")
    st.markdown(learn_more)
    st.markdown("### Additional Resources")
    st.markdown("- [Psychology Today: The Dark Triad](https://www.psychologytoday.com/us/basics/dark-triad)")
    st.markdown("- [Article: The Dark Triad in Everyday Life](https://www.apa.org/monitor/2016/12/dark-triad)")
    st.markdown("- [ResearchGate: Introducing a Short Measure of the Dark Triad](https://www.researchgate.net/publication/271258362_Introducing_a_short_measure_of_the_Dark_Triad)")

with tab3:
    st.header("Need Help? Resources and Helplines")
    st.markdown(helpline_info)
    st.markdown("### More Help")
    st.markdown("If you feel overwhelmed or need support, please consider reaching out to a trusted professional or local support service.")
    st.markdown("[Find international helpline resources here](https://findahelpline.com/)")