import streamlit as st

# Set page config FIRST - before any other streamlit imports
st.set_page_config(
    page_title="PersonaPredict Pro | Advanced Personality Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other libraries that might use streamlit
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import time
import plotly.graph_objects as go
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

# Custom CSS for professional styling
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
    /* Custom toggle switch */
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
    /* Progress bar */
    .stProgress>div>div>div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
    }
    /* Custom metric cards */
    .stMetric {
        border-radius: 12px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05) !important;
    }
    /* Custom tabs */
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
    /* Custom form */
    .stForm {
        border: 1px solid #e9ecef;
        border-radius: 16px;
        padding: 2rem;
        background: var(--lighter);
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
    }
    /* Custom divider */
    .stDivider>div>div>div {
        background: linear-gradient(90deg, var(--primary), var(--accent));
        height: 3px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model (with error handling)
@st.cache_resource
def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to create interactive gauge chart
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

# Function to create personality radar chart
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

# Main app function
def main():
    # Main content area 
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
        st.image("66098.png", width=250)
    st.markdown("---")

    # Load model
    model = load_model("personality_model.joblib")
    if model is None:
        st.error("Model could not be loaded. Please check the model file.")
        return

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

    # Create form for user input
    with st.form("personality_form"):
        st.markdown("<h2 class='subheader'>Behavioral Questionnaire</h2>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color: #495057; margin-bottom: 1.5rem;'>
        Please answer the following questions honestly about your typical behaviors and preferences.
        There are no right or wrong answers - this is about understanding your natural tendencies.
        </p>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            with stylable_container(
                key="social_container",
                css_styles="""
                {
                    background-color: #f8f9fa;
                    border-radius: 16px;
                    padding: 1.5rem;
                    border: 1px solid #e9ecef;
                    margin-right: 0rem !important;
                }
                """
            ):
                st.markdown("<h4 style='color: #4a6bff; margin-bottom: 1rem;'>Social Behavior Patterns</h4>", unsafe_allow_html=True)
                time_alone = st.slider(
                    "Time spent alone per day (hours)", 
                    0, 11, 5,
                    help="Average hours spent alone each day"
                )
                social_events = st.slider(
                    "Social events attended per month", 
                    0, 10, 3,
                    help="How many social events you typically attend each month"
                )
                going_outside = st.slider(
                    "Times going outside per week", 
                    0, 7, 4,
                    help="How many times you leave your home each week"
                )
                friends_circle = st.slider(
                    "Close friends circle size", 
                    0, 15, 5,
                    help="Number of people you consider close friends"
                )
        with col2:
            with stylable_container(
                key="psych_container",
                css_styles="""
                {
                    background-color: #f8f9fa;
                    border-radius: 16px;
                    padding: 1.5rem;
                    border: 1px solid #e9ecef;
                    margin-right: 0rem !important;
                }
                """
            ):
                st.markdown("<h4 style='color: #4a6bff; margin-bottom: 1rem;'>Psychological & Digital Factors</h4>", unsafe_allow_html=True)
                stage_fear = st.selectbox(
                    "Do you experience stage fear?",
                    [("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    help="Do you feel anxious about public speaking or performing?"
                )[1]
                drained = st.selectbox(
                    "Do you feel drained after socializing?",
                    [("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    help="Do you need alone time to recharge after social interactions?"
                )[1]
                post_freq = st.slider(
                    "Social media posts per week", 
                    0, 10, 2,
                    help="How often you post on social media each week"
                )
                high_engagement = st.selectbox(
                    "Do you actively engage in conversations?",
                    [("No", 0), ("Yes", 1)],
                    format_func=lambda x: x[0],
                    help="Do you frequently initiate or actively participate in discussions?"
                )[1]
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            submitted = st.form_submit_button("Analyze My Personality", 
                                            use_container_width=True,
                                            type="primary")

    # When form is submitted
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame([[time_alone, stage_fear, social_events, going_outside, 
                                   drained, friends_circle, post_freq, high_engagement]],
                                columns=feature_names)
        # Make prediction
        try:
            with st.spinner('Analyzing your personality profile...'):
                progress_bar = st.progress(0, text="Processing your responses")
                for percent_complete in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(percent_complete + 1, text="Processing your responses")
                time.sleep(0.1)
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0]
                # For 1=Introvert and 0=Extrovert
                introvert_prob = proba[1]
                extrovert_prob = proba[0]
                # Display results
                st.markdown("---")
                st.markdown("<h2 class='header'>Your Personality Profile Results</h2>", unsafe_allow_html=True)
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    with stylable_container(
                        key="metric_primary",
                        css_styles="""
                        {
                            background-color: white;
                            border-radius: 12px;
                            padding: 1rem;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                            border-left: 4px solid #4a6bff;
                        }
                        """
                    ):
                        st.metric("Primary Tendency", 
                                 "Introvert" if prediction == 1 else "Extrovert",
                                 f"{'Introvert' if prediction == 1 else 'Extrovert'} tendency detected")
                with col2:
                    with stylable_container(
                        key="metric_intro",
                        css_styles="""
                        {
                            background-color: white;
                            border-radius: 12px;
                            padding: 1rem;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                            border-left: 4px solid #6a5acd;
                        }
                        """
                    ):
                        st.metric("Introversion Score", 
                                 f"{introvert_prob*100:.1f}%",
                                 "Higher means more introverted")
                with col3:
                    with stylable_container(
                        key="metric_extra",
                        css_styles="""
                        {
                            background-color: white;
                            border-radius: 12px;
                            padding: 1rem;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                            border-left: 4px solid #ff6b6b;
                        }
                        """
                    ):
                        st.metric("Extroversion Score", 
                                 f"{extrovert_prob*100:.1f}%",
                                 "Higher means more extroverted")
                style_metric_cards(background_color="#FFFFFF", border_left_color="#4a6bff", box_shadow="0 4px 8px rgba(0,0,0,0.05)")
                # Main results section
                tab1, tab2, tab3 = st.tabs(["Personality Spectrum", "Profile Analysis", "Behavioral Patterns"])
                with tab1:
                    col_res1, col_res2 = st.columns([1, 1])
                    with col_res1:
                        st.markdown("<h4 class='section-title'>Your Personality Spectrum</h4>", unsafe_allow_html=True)
                        gauge_fig = create_gauge_chart(introvert_prob, "Introversion Level")
                        st.plotly_chart(gauge_fig, use_container_width=True)
                    # Behavioral Radar Chart removed
                with tab2:
                    with stylable_container(
                        key="prediction_card",
                        css_styles="""
                        {
                            background-color: white;
                            border-radius: 16px;
                            padding: 2rem;
                            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
                            border-left: 6px solid #4a6bff;
                        }
                        """
                    ):
                        if prediction == 1:
                            st.markdown("""
                            <div style='color: #2b2d42;'>
                            <h2 style='color: #4a6bff; margin-bottom: 1rem;'>🧠 Introvert Personality Profile</h2>
                            <p style='margin-bottom: 1rem; line-height: 1.6;'>
                            Your responses indicate a <strong>strong preference for introverted tendencies</strong>. 
                            This psychological profile suggests you likely:
                            </p>
                            <ul style='margin-bottom: 1.5rem; padding-left: 1.5rem; line-height: 1.6;'>
                                <li style='margin-bottom: 0.5rem;'>Recharge energy through solitude and quiet environments</li>
                                <li style='margin-bottom: 0.5rem;'>Prefer deep, meaningful conversations over small talk</li>
                                <li style='margin-bottom: 0.5rem;'>Experience mental fatigue after extensive social interaction</li>
                                <li style='margin-bottom: 0.5rem;'>Maintain a selective, close-knit social circle</li>
                                <li style='margin-bottom: 0.5rem;'>Process information internally before speaking</li>
                            </ul>
                            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #6a5acd;'>
                                <h4 style='color: #4a6bff; margin-bottom: 0.5rem;'>Professional Insight</h4>
                                <p style='margin-bottom: 0; line-height: 1.6; font-style: italic;'>
                                Introversion is about energy management, not social ability. Many successful leaders and 
                                innovators are introverts who leverage their reflective nature, deep focus, and 
                                thoughtful decision-making. Your preference for meaningful interaction can be a 
                                strength in building authentic professional relationships.
                                </p>
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='color: #2b2d42;'>
                            <h2 style='color: #4a6bff; margin-bottom: 1rem;'>🧠 Extrovert Personality Profile</h2>
                            <p style='margin-bottom: 1rem; line-height: 1.6;'>
                            Your responses indicate a <strong>strong preference for extroverted tendencies</strong>. 
                            This psychological profile suggests you likely:
                            </p>
                            <ul style='margin-bottom: 1.5rem; padding-left: 1.5rem; line-height: 1.6;'>
                                <li style='margin-bottom: 0.5rem;'>Gain energy from social interaction and external stimulation</li>
                                <li style='margin-bottom: 0.5rem;'>Enjoy being around people and group activities</li>
                                <li style='margin-bottom: 0.5rem;'>Think out loud and process information through conversation</li>
                                <li style='margin-bottom: 0.5rem;'>Maintain a broad network of acquaintances</li>
                                <li style='margin-bottom: 0.5rem;'>Feel comfortable in dynamic, interactive settings</li>
                            </ul>
                            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff6b6b;'>
                                <h4 style='color: #4a6bff; margin-bottom: 0.5rem;'>Professional Insight</h4>
                                <p style='margin-bottom: 0; line-height: 1.6; font-style: italic;'>
                                Extroversion is about energy sources, not just sociability. Many creative professionals 
                                and leaders are extroverts who thrive on collaboration, brainstorming, and team dynamics. 
                                Your natural inclination toward interaction can be powerful for networking, team building, 
                                and motivating others.
                                </p>
                            </div>
                            </div>
                            """, unsafe_allow_html=True)
                with tab3:
                    st.markdown("<h4 class='section-title'>Your Behavioral Patterns</h4>", unsafe_allow_html=True)
                    col_beh1, col_beh2 = st.columns(2)
                    with col_beh1:
                        with stylable_container(
                            key="behavior_card1",
                            css_styles="""
                            {
                                background-color: white;
                                border-radius: 16px;
                                padding: 1.5rem;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                            }
                            """
                        ):
                            st.markdown("<h5 style='color: #4a6bff; margin-bottom: 1rem;'>Social Engagement</h5>", unsafe_allow_html=True)
                            st.metric("Social Events/Month", social_events, 
                                     "Higher indicates more extroverted" if social_events > 5 else "Lower indicates more introverted")
                            st.metric("Close Friends", friends_circle, 
                                     "Larger circles suggest extroversion" if friends_circle > 8 else "Smaller circles suggest introversion")
                            st.metric("Conversation Engagement", "High" if high_engagement else "Low", 
                                     "Active engagement suggests extroversion" if high_engagement else "Less engagement suggests introversion")
                    with col_beh2:
                        with stylable_container(
                            key="behavior_card2",
                            css_styles="""
                            {
                                background-color: white;
                                border-radius: 16px;
                                padding: 1.5rem;
                                box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                            }
                            """
                        ):
                            st.markdown("<h5 style='color: #4a6bff; margin-bottom: 1rem;'>Psychological Factors</h5>", unsafe_allow_html=True)
                            st.metric("Stage Fear", "Yes" if stage_fear else "No", 
                                     "Common in introverts" if stage_fear else "Less common in extroverts")
                            st.metric("Post-Social Energy", "Drained" if drained else "Energized", 
                                     "Typical of introverts" if drained else "Typical of extroverts")
                            st.metric("Social Media Activity", f"{post_freq} posts/week", 
                                     "Higher activity suggests extroversion" if post_freq > 5 else "Lower activity suggests introversion")
                # Recommendations section removed
                st.markdown("---")
                results_str = f"""PERSONALITY PROFILE REPORT - PERSONAPREDICT PRO

Primary Tendency: {'Introvert' if prediction == 1 else 'Extrovert'}
Introversion Score: {introvert_prob*100:.1f}%
Extroversion Score: {extrovert_prob*100:.1f}%

BEHAVIORAL PATTERNS:
- Time spent alone: {time_alone} hours/day
- Social events attended: {social_events} times/month
- Times going outside: {going_outside} times/week
- Close friends circle: {friends_circle} people
- Social media posts: {post_freq} times/week

PSYCHOLOGICAL FACTORS:
- Stage fear: {'Yes' if stage_fear == 1 else 'No'}
- Drained after socializing: {'Yes' if drained == 1 else 'No'}
- Actively engages in conversations: {'Yes' if high_engagement == 1 else 'No'}

Generated by PersonaPredict Pro - {time.strftime("%Y-%m-%d")}
"""
                col_dl1, col_dl2, col_dl3 = st.columns([1,2,1])
                with col_dl2:
                    st.download_button(
                        label="📄 Download Full Report (PDF)",
                        data=results_str,
                        file_name="persona_predict_profile_report.txt",
                        mime="text/plain",
                        use_container_width=True,
                        type="secondary"
                    )
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    main()
    # Add creator profile links at the end of the app
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 1.1rem; margin-top: 2rem;'>
            <span>Connect with the app creator:</span><br>
            <a href="https://www.linkedin.com/in/furqan-khan-256798268/" target="_blank" style="margin-right: 20px;">
                <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg" alt="LinkedIn" width="28" style="vertical-align:middle; margin-right:8px;">LinkedIn
            </a>
            <a href="https://github.com/furqank73" target="_blank">
                <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/github.svg" alt="GitHub" width="28" style="vertical-align:middle; margin-right:8px;">GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )