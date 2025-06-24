# ✅ MUST BE FIRST: Streamlit import + config
import streamlit as st

st.set_page_config(
    page_title="PersonaPredict Pro | Advanced Personality Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Other imports (AFTER Streamlit config)
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container

# ====================== CSS Styling ======================
st.markdown("""
    <style>
    :root {
        --primary: #4a6bff;
        --secondary: #6a5acd;
        --accent: #ff6b6b;
        --dark: #2b2d42;
        --darker: #1a1a2e;
        --light: #f8f9fa;
        --lighter: #ffffff;
        --success: #38b000;
        --warning: #ffaa00;
        --danger: #ef233c;
        --info: #00b4d8;
    }
    .main {
        background-color: var(--light);
    }
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, var(--secondary), var(--primary));
    }
    .stSlider>div>div>div>div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    .stRadio>div {
        background-color: var(--lighter);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .stRadio>div>label>div:first-child {
        background-color: var(--lighter) !important;
    }
    .stRadio>div>div:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        background: var(--lighter);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-left: 6px solid var(--primary);
        transition: transform 0.3s ease;
    }
    .prediction-card:hover {
        transform: translateY(-5px);
    }
    .feature-card {
        padding: 1.5rem;
        background: var(--lighter);
        border-radius: 16px;
        margin-top: 1.5rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-top: 4px solid var(--secondary);
    }
    .header {
        color: var(--dark);
        font-weight: 800;
        margin-bottom: 0.75rem;
        font-size: 2.5rem;
    }
    .subheader {
        color: var(--primary);
        font-weight: 700;
        margin-bottom: 0.75rem;
        font-size: 1.5rem;
    }
    .section-title {
        color: var(--dark);
        font-weight: 700;
        margin-bottom: 1rem;
        font-size: 1.25rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 12px;
        border: 1px solid #dee2e6;
    }
    .stSelectbox>div>div>div>div {
        border-radius: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .css-1aumxhk {
        background-color: var(--light);
    }
    .stRadio [role="radiogroup"] {
        gap: 1rem;
    }
    .stRadio [role="radio"] {
        padding: 0.75rem 1rem;
        border-radius: 10px;
        background: var(--lighter);
        border: 1px solid #e9ecef !important;
    }
    .stRadio [role="radio"][aria-checked="true"] {
        background: var(--primary) !important;
        color: white !important;
        border: 1px solid var(--primary) !important;
        box-shadow: 0 2px 8px rgba(74, 107, 255, 0.3);
    }
    .stProgress>div>div>div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    .stMetric {
        border-radius: 12px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important;
    }
    .stTabs [role="tablist"] {
        gap: 0.5rem;
    }
    .stTabs [role="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: 12px 12px 0 0;
        background: #f1f3f5;
        color: var(--dark);
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: var(--primary) !important;
        color: white !important;
    }
    .stTabs [role="tab"]:hover {
        background: #e9ecef;
    }
    .stForm {
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 2rem;
        background: var(--lighter);
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
    }
    .stDivider>div>div>div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
        height: 3px !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== Helper Functions ======================
def create_gauge_chart(probability, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability*100,
        number = {'suffix': "%", 'font': {'size': 36}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20, 'color': '#2b2d42'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#4a6bff"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d0d5ff'},
                {'range': [30, 70], 'color': '#a5b4fc'},
                {'range': [70, 100], 'color': '#4a6bff'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': probability*100}}))
    fig.update_layout(
        height=350,
        margin=dict(l=0, r=0, b=0, t=40, pad=0),
        font=dict(color="#2b2d42", family="Arial"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_radar_chart(features, values):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=features,
        fill='toself',
        name='Your Scores',
        line_color='#4a6bff',
        fillcolor='rgba(74, 107, 255, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values)*1.2],
                color='#6c757d'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=350,
        margin=dict(l=50, r=50, b=50, t=50, pad=0),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#2b2d42", family="Arial")
    )
    return fig

@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ====================== Main App ======================
def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h1 class='header'>Advanced Personality Spectrum Analysis</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #495057; font-size: 1.1rem; line-height: 1.6;'>
        Discover where you fall on the introversion-extroversion spectrum through our scientifically validated 
        behavioral assessment. Gain insights into your social preferences, energy sources, and communication style.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        st.image("https://via.placeholder.com/250x150?text=Personality+AI", width=250)  # Replace with your image
    st.markdown("---")

    # Load model
    model = load_model("personality_model.joblib")
    if model is None:
        st.stop()

    # Feature names
    feature_names = [
        'Time_spent_Alone',
        'Stage_fear',
        'Social_event_attendance',
        'Going_outside',
        'Drained_after_socializing',
        'Friends_circle_size',
        'Post_frequency',
        'High_Engagement'
    ]

    # Input form
    with st.form("personality_form"):
        st.markdown("<h2 class='subheader'>Behavioral Questionnaire</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #495057; margin-bottom: 1.5rem;'>
        Please answer the following questions honestly about your typical behaviors and preferences.
        There are no right or wrong answers - this is about understanding your natural tendencies.
        </p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            with stylable_container(key="social_container", css_styles="""
                { background-color: #f8f9fa; border-radius: 16px; padding: 1.5rem; border: 1px solid #e9ecef; }
                """):
                st.markdown("<h4 style='color: #4a6bff; margin-bottom: 1rem;'>Social Behavior Patterns</h4>", unsafe_allow_html=True)
                time_alone = st.slider("Time spent alone per day (hours)", 0, 11, 5)
                social_events = st.slider("Social events attended per month", 0, 10, 3)
                going_outside = st.slider("Times going outside per week", 0, 7, 4)
                friends_circle = st.slider("Close friends circle size", 0, 15, 5)
        
        with col2:
            with stylable_container(key="psych_container", css_styles="""
                { background-color: #f8f9fa; border-radius: 16px; padding: 1.5rem; border: 1px solid #e9ecef; }
                """):
                st.markdown("<h4 style='color: #4a6bff; margin-bottom: 1rem;'>Psychological & Digital Factors</h4>", unsafe_allow_html=True)
                stage_fear = st.selectbox("Do you experience stage fear?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
                drained = st.selectbox("Do you feel drained after socializing?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
                post_freq = st.slider("Social media posts per week", 0, 10, 2)
                high_engagement = st.selectbox("Do you actively engage in conversations?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        
        submitted = st.form_submit_button("Analyze My Personality", use_container_width=True, type="primary")

    # Prediction
    if submitted:
        input_data = pd.DataFrame([[time_alone, stage_fear, social_events, going_outside, 
                                 drained, friends_circle, post_freq, high_engagement]],
                                columns=feature_names)
        try:
            with st.spinner('Analyzing your personality profile...'):
                progress_bar = st.progress(0, text="Processing your responses")
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1)
                time.sleep(0.1)
                
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                introvert_prob = proba[1]
                extrovert_prob = proba[0]
                
                # Results display
                st.markdown("---")
                st.markdown("<h2 class='header'>Your Personality Profile Results</h2>", unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    with stylable_container(key="metric_primary", css_styles="""
                        { background-color: white; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05); border-left: 4px solid #4a6bff; }
                        """):
                        st.metric("Primary Tendency", "Introvert" if prediction == 1 else "Extrovert")
                with col2:
                    with stylable_container(key="metric_intro", css_styles="""
                        { background-color: white; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05); border-left: 4px solid #6a5acd; }
                        """):
                        st.metric("Introversion Score", f"{introvert_prob*100:.1f}%")
                with col3:
                    with stylable_container(key="metric_extra", css_styles="""
                        { background-color: white; border-radius: 12px; padding: 1rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05); border-left: 4px solid #ff6b6b; }
                        """):
                        st.metric("Extroversion Score", f"{extrovert_prob*100:.1f}%")
                
                style_metric_cards(background_color="#FFFFFF", border_left_color="#4a6bff", box_shadow="0 4px 8px rgba(0,0,0,0.05)")
                
                # Tabs
                tab1, tab2, tab3 = st.tabs(["Personality Spectrum", "Profile Analysis", "Behavioral Patterns"])
                
                with tab1:
                    st.plotly_chart(create_gauge_chart(introvert_prob, "Introversion Level"), use_container_width=True)
                
                with tab2:
                    with stylable_container(key="prediction_card", css_styles="""
                        { background-color: white; border-radius: 16px; padding: 2rem; box-shadow: 0 8px 16px rgba(0,0,0,0.08); border-left: 6px solid #4a6bff; }
                        """):
                        if prediction == 1:
                            st.markdown("""
                            <div style='color: #2b2d42;'>
                            <h2 style='color: #4a6bff; margin-bottom: 1rem;'>🧠 Introvert Personality Profile</h2>
                            <p style='margin-bottom: 1rem; line-height: 1.6;'>
                            Your responses indicate a <strong>strong preference for introverted tendencies</strong>.
                            </p>
                            <ul style='margin-bottom: 1.5rem; padding-left: 1.5rem; line-height: 1.6;'>
                                <li>Recharge energy through solitude</li>
                                <li>Prefer deep conversations</li>
                                <li>Experience mental fatigue after socializing</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='color: #2b2d42;'>
                            <h2 style='color: #4a6bff; margin-bottom: 1rem;'>🧠 Extrovert Personality Profile</h2>
                            <p style='margin-bottom: 1rem; line-height: 1.6;'>
                            Your responses indicate a <strong>strong preference for extroverted tendencies</strong>.
                            </p>
                            <ul style='margin-bottom: 1.5rem; padding-left: 1.5rem; line-height: 1.6;'>
                                <li>Gain energy from social interaction</li>
                                <li>Enjoy group activities</li>
                                <li>Think out loud</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                
                with tab3:
                    col_beh1, col_beh2 = st.columns(2)
                    with col_beh1:
                        with stylable_container(key="behavior_card1", css_styles="""
                            { background-color: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
                            """):
                            st.markdown("<h5 style='color: #4a6bff; margin-bottom: 1rem;'>Social Engagement</h5>", unsafe_allow_html=True)
                            st.metric("Social Events/Month", social_events)
                            st.metric("Close Friends", friends_circle)
                    with col_beh2:
                        with stylable_container(key="behavior_card2", css_styles="""
                            { background-color: white; border-radius: 16px; padding: 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
                            """):
                            st.markdown("<h5 style='color: #4a6bff; margin-bottom: 1rem;'>Psychological Factors</h5>", unsafe_allow_html=True)
                            st.metric("Stage Fear", "Yes" if stage_fear else "No")
                            st.metric("Post-Social Energy", "Drained" if drained else "Energized")
                
                # Download report
                results_str = f"""PERSONALITY PROFILE REPORT

Primary Tendency: {'Introvert' if prediction == 1 else 'Extrovert'}
Introversion Score: {introvert_prob*100:.1f}%
Extroversion Score: {extrovert_prob*100:.1f}%

Generated on {time.strftime("%Y-%m-%d")}
"""
                st.download_button(
                    label="📄 Download Report",
                    data=results_str,
                    file_name="personality_profile.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; margin-top: 2rem;'>
        <p>PersonaPredict Pro © 2024 | Advanced Personality Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()