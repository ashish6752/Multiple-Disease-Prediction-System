import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu


# ================== Load Models ==================
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))


# ================== Helper: safe numeric conversion ==================
def to_float_row(labels_and_values):
    """
    Converts a list of (label, raw_text) pairs into a list of floats.
    Returns (values, errors) where errors is a list of human-readable
    messages for any field that was blank or not a valid number.
    """
    values = []
    errors = []
    for label, raw in labels_and_values:
        raw = (raw or "").strip()
        if raw == "":
            errors.append(f"'{label}' is empty.")
            continue
        try:
            values.append(float(raw))
        except ValueError:
            errors.append(f"'{label}' must be a number (got '{raw}').")
    return values, errors


# ================== Background Image ==================
def set_bg_from_url(url, opacity=1):
    st.markdown(
        f"""
        <style>
        body {{
            background: url("{url}") no-repeat center center fixed;
            background-size: cover;
            opacity: {opacity};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_url(
    "https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768",
    opacity=0.9
)


# ================== Sidebar Menu ==================
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction',
         'Heart Disease Prediction',
         'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

st.title("AI Driven Risk Analysis of Healthcare Data for Precision Medicine in Chronic Disease")


# ================== Diabetes Prediction ==================
if selected == 'Diabetes Prediction':

    st.subheader('Diabetes Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        SkinThickness = st.text_input('Skin Thickness')
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

    with col2:
        Glucose = st.text_input('Glucose Level')
        Insulin = st.text_input('Insulin Level')
        Age = st.text_input('Age')

    with col3:
        BloodPressure = st.text_input('Blood Pressure')
        BMI = st.text_input('BMI')

    if st.button('Diabetes Test Result'):
        fields = [
            ('Number of Pregnancies', Pregnancies),
            ('Glucose Level', Glucose),
            ('Blood Pressure', BloodPressure),
            ('Skin Thickness', SkinThickness),
            ('Insulin Level', Insulin),
            ('BMI', BMI),
            ('Diabetes Pedigree Function', DiabetesPedigreeFunction),
            ('Age', Age),
        ]
        values, errors = to_float_row(fields)

        if errors:
            st.error("Please fix the following before predicting:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            diab_prediction = diabetes_model.predict(np.array([values]))

            if diab_prediction[0] == 1:
                st.warning("⚠️ The person is diabetic")

                st.markdown("### 🩺 Doctor Consultation")
                st.info("Consult an endocrinologist or physician for further evaluation.")

                st.markdown("### 🥗 Diet Recommendations")
                st.markdown("""
                - Whole grains and green vegetables  
                - Avoid sugar and sweets  
                - Low-carb, high-fiber foods  
                - Drink plenty of water
                """)

                st.markdown("### 🏃 Physical Exercises")
                st.markdown("""
                - Brisk walking (30 minutes daily)  
                - Yoga and stretching  
                - Light strength training  
                - Avoid sedentary lifestyle
                """)

            else:
                st.success("✅ The person is not diabetic")
                st.info("Maintain a healthy lifestyle and regular checkups.")


# ================== Heart Disease Prediction ==================
if selected == 'Heart Disease Prediction':

    st.subheader('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
        trestbps = st.text_input('Resting Blood Pressure')
        restecg = st.text_input('Resting ECG')
        oldpeak = st.text_input('ST Depression')

    with col2:
        sex = st.text_input('Sex (1 = male, 0 = female)')
        chol = st.text_input('Serum Cholesterol')
        thalach = st.text_input('Max Heart Rate')
        slope = st.text_input('Slope')

    with col3:
        cp = st.text_input('Chest Pain Type')
        fbs = st.text_input('Fasting Blood Sugar')
        exang = st.text_input('Exercise Induced Angina')
        ca = st.text_input('Major Vessels')
        thal = st.text_input('Thal')

    if st.button('Heart Disease Test Result'):
        fields = [
            ('Age', age),
            ('Sex', sex),
            ('Chest Pain Type', cp),
            ('Resting Blood Pressure', trestbps),
            ('Serum Cholesterol', chol),
            ('Fasting Blood Sugar', fbs),
            ('Resting ECG', restecg),
            ('Max Heart Rate', thalach),
            ('Exercise Induced Angina', exang),
            ('ST Depression', oldpeak),
            ('Slope', slope),
            ('Major Vessels', ca),
            ('Thal', thal),
        ]
        values, errors = to_float_row(fields)

        if errors:
            st.error("Please fix the following before predicting:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            heart_prediction = heart_disease_model.predict(np.array([values]))

            if heart_prediction[0] == 1:
                st.warning("⚠️ The person has heart disease")

                st.markdown("### 🩺 Doctor Consultation")
                st.info("Consult a cardiologist immediately.")

                st.markdown("### 🥗 Diet Recommendations")
                st.markdown("""
                - Low salt & low fat foods  
                - Avoid fried and processed foods  
                - Eat fruits, vegetables, nuts  
                - Reduce cholesterol intake
                """)

                st.markdown("### ❤️ Physical Activities")
                st.markdown("""
                - Walking or cycling  
                - Light aerobic exercises  
                - Stress management & meditation  
                - Avoid smoking and alcohol
                """)

            else:
                st.success("✅ The person does not have heart disease")
                st.info("Continue healthy diet and regular exercise.")


# ================== Parkinson's Prediction ==================
if selected == "Parkinsons Prediction":

    st.subheader("Parkinson's Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        RAP = st.text_input('MDVP:RAP')
        APQ3 = st.text_input('Shimmer:APQ3')
        NHR = st.text_input('NHR')
        spread1 = st.text_input('spread1')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        PPQ = st.text_input('MDVP:PPQ')
        APQ5 = st.text_input('Shimmer:APQ5')
        HNR = st.text_input('HNR')
        spread2 = st.text_input('spread2')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        APQ = st.text_input('MDVP:APQ')
        RPDE = st.text_input('RPDE')
        D2 = st.text_input('D2')

    with col4:
        Jitter_percent = st.text_input('Jitter (%)')
        Shimmer = st.text_input('MDVP:Shimmer')
        DDA = st.text_input('Shimmer:DDA')
        DFA = st.text_input('DFA')
        PPE = st.text_input('PPE')

    with col5:
        Jitter_Abs = st.text_input('Jitter (Abs)')
        Shimmer_dB = st.text_input('Shimmer(dB)')

    if st.button("Parkinson's Test Result"):
        fields = [
            ('MDVP:Fo(Hz)', fo),
            ('MDVP:Fhi(Hz)', fhi),
            ('MDVP:Flo(Hz)', flo),
            ('Jitter (%)', Jitter_percent),
            ('Jitter (Abs)', Jitter_Abs),
            ('MDVP:RAP', RAP),
            ('MDVP:PPQ', PPQ),
            ('Jitter:DDP', DDP),
            ('MDVP:Shimmer', Shimmer),
            ('Shimmer(dB)', Shimmer_dB),
            ('Shimmer:APQ3', APQ3),
            ('Shimmer:APQ5', APQ5),
            ('MDVP:APQ', APQ),
            ('Shimmer:DDA', DDA),
            ('NHR', NHR),
            ('HNR', HNR),
            ('RPDE', RPDE),
            ('DFA', DFA),
            ('spread1', spread1),
            ('spread2', spread2),
            ('D2', D2),
            ('PPE', PPE),
        ]
        values, errors = to_float_row(fields)

        if errors:
            st.error("Please fix the following before predicting:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            parkinsons_prediction = parkinsons_model.predict(np.array([values]))

            if parkinsons_prediction[0] == 1:
                st.warning("⚠️ The person has Parkinson's disease")

                st.markdown("### 🩺 Doctor Consultation")
                st.info("Consult a neurologist for diagnosis and treatment.")

                st.markdown("### 🥗 Diet Advice")
                st.markdown("""
                - High-fiber foods  
                - Antioxidant-rich fruits  
                - Adequate hydration  
                - Avoid processed foods
                """)

                st.markdown("### 🧘 Physical & Speech Exercises")
                st.markdown("""
                - Physiotherapy  
                - Balance exercises  
                - Speech therapy  
                - Daily stretching
                """)

            else:
                st.success("✅ The person does not have Parkinson's disease")
                st.info("Maintain an active and healthy lifestyle.")