import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def load_model(model_path='/Users/hp/Desktop/atul-webpage/Newfolder/New_folder/trained.h5'):
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)

def preprocess_image(img, target_size=(300, 300)):
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_prescription_and_advice(prediction, age, symptoms):
    recommendations = {
        "prescription": [],
        "advice": [],
        "severity": "mild" if prediction < 0.7 else "moderate" if prediction < 0.85 else "severe"
    }
    
    # Basic prescription recommendations based on severity
    if recommendations["severity"] == "mild":
        recommendations["prescription"] = [
            "Amoxicillin 500mg twice daily for 7 days",
            "Acetaminophen 500mg for fever and pain as needed",
            "Dextromethorphan for cough suppression"
        ]
    elif recommendations["severity"] == "moderate":
        recommendations["prescription"] = [
            "Azithromycin 500mg once daily for 5 days",
            "Ibuprofen 400mg for inflammation and fever",
            "Codeine-based cough suppressant",
            "Consider bronchodilator inhaler"
        ]
    else:  # severe
        recommendations["prescription"] = [
            "Immediate hospitalization may be required",
            "Broad-spectrum antibiotics (hospital administered)",
            "Oxygen therapy as needed",
            "Intensive monitoring recommended"
        ]
    
    # General advice based on symptoms and severity
    recommendations["advice"] = [
        "Rest and stay hydrated",
        "Monitor temperature regularly",
        "Use a humidifier to ease breathing",
        "Practice deep breathing exercises when possible"
    ]
    
    if symptoms['Fever']:
        recommendations["advice"].append("Take lukewarm baths to manage fever")
    
    if symptoms['Breathing Difficulty']:
        recommendations["advice"].append("Sleep in an elevated position with extra pillows")
        recommendations["advice"].append("Avoid strenuous activities")
    
    if age > 65:
        recommendations["advice"].append("Extra monitoring required due to age-related risks")
        
    return recommendations

def main():
    st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")
    st.title("🫁 Pneumonia Detection System")
    st.markdown("#### *Empowering Healthcare with AI: Early Detection Saves Lives*")
    st.markdown("---")
    
    model = load_model()
    
    if model is None:
        st.error("Error: Model file not found!")
        return

    with st.sidebar:
        st.header("Patient Information")
        age = st.number_input("Age", 0, 120, 25)
        symptoms = {
            'Fever': st.checkbox("Fever"),
            'Cough': st.checkbox("Cough"),
            'Breathing Difficulty': st.checkbox("Difficulty Breathing"),
            'Chest Pain': st.checkbox("Chest Pain")
        }

    st.header("X-ray Analysis")
    uploaded_file = st.file_uploader("Upload chest X-ray", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-ray", use_container_width=True)
        
        if st.button("Analyze X-ray"):
            with st.spinner("Processing..."):
                try:
                    processed_img = preprocess_image(img)
                    prediction = model.predict(processed_img, verbose=0)[0][0]
                    
                    if prediction > 0.5:
                        st.error(f"Pneumonia detected: {prediction:.2%} probability")
                        
                        # Get recommendations based on prediction
                        recommendations = get_prescription_and_advice(prediction, age, symptoms)
                        
                        # Display recommendations in an expander
                        with st.expander("📋 View Recommendations", expanded=True):
                            st.subheader("🏥 Medical Recommendations")
                            st.warning("⚠️ These are AI-generated suggestions. Always consult a healthcare provider for proper diagnosis and treatment.")
                            
                            st.markdown(f"**Severity Level**: {recommendations['severity'].title()}")
                            
                            st.markdown("**💊 Suggested Prescription:**")
                            for med in recommendations['prescription']:
                                st.markdown(f"- {med}")
                            
                            st.markdown("**🏡 Home Care Advice:**")
                            for advice in recommendations['advice']:
                                st.markdown(f"- {advice}")
                            
                    else:
                        st.success(f"No pneumonia detected: {(1-prediction):.2%} probability")
                        st.info("Continue monitoring symptoms and consult a healthcare provider if conditions worsen.")
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()