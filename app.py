"""
app.py — Premium Titanic AI Analyst

A high-performance, robust frontend using native Streamlit chat components 
for the best user experience.
"""

import os
import requests
import streamlit as st
import seaborn as sns

# ──────────────────────────────────────────────
# 1. CORE CONFIGURATION
# ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Titanic AI Explorer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 2. PREMIUM STYLING (Fixed Sidebar & Chat)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global Reset */
    .main { background-color: #0f0f1a; }
    
    /* Sidebar Force Visibility & Color */
    [data-testid="stSidebar"] {
        background-image: linear-gradient(#14142b, #0d0d16) !important;
        min-width: 300px !important;
    }
    
    /* Smooth Transitions for Chat */
    .stChatMessage {
        animation: fadeIn 0.5s;
        border: 1px solid rgba(255,255,255,0.05) !important;
        background: rgba(255,255,255,0.02) !important;
        margin-bottom: 10px !important;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Style Suggestion Buttons */
    div.stButton > button {
        border-radius: 20px !important;
        border: 1px solid #6c63ff !important;
        color: #fff !important;
        background: transparent !important;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: #6c63ff !important;
        box-shadow: 0 0 15px rgba(108, 99, 255, 0.4);
    }
    
    /* KPI Styling */
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #6c63ff;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 3. SIDEBAR (Explorer Panel)
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧭 Navigator")
    st.info("The sidebar is your command center. Check vessel stats here.")
    
    # Load Data for Stats
    @st.cache_data
    def load_data():
        return sns.load_dataset("titanic")
    
    df = load_data()
    
    st.markdown("### Vessel Stats")
    cols = st.columns(1)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><small>TOTAL PASSENGERS</small><br><b>{len(df)}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><small>SURVIVAL RATE</small><br><b>{df["survived"].mean()*100:.1f}%</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><small>AVG FARE Paid</small><br><b>${df["fare"].mean():.2f}</b></div>', unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Reset Agent", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────
# 4. MAIN INTERFACE
# ──────────────────────────────────────────────
st.title("🚢 Titanic AI Analyst")
st.caption("Ask our AI anything about the Titanic dataset — including charts and deep analysis.")

# Session State for Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "chart" in msg and msg["chart"]:
            if os.path.exists(msg["chart"]):
                st.image(msg["chart"])

# Quick Suggestions (Interactive)
st.write("---")
st.write("💡 **Try asking:**")
sug_cols = st.columns(2)
with sug_cols[0]:
    if st.button("📊 Show Age Distribution", use_container_width=True):
        selected_prompt = "Show the distribution of passenger ages"
    if st.button("⚓ Port of Embarkation stats", use_container_width=True):
        selected_prompt = "Show passengers per embarkation port"
with sug_cols[1]:
    if st.button("🛟 What was the survival rate?", use_container_width=True):
        selected_prompt = "What was the overall survival rate?"
    if st.button("💰 Class vs Fare analysis", use_container_width=True):
        selected_prompt = "What was the average fare for each class?"

# Chat Input (at the bottom)
query = st.chat_input("Ask a question about the Titanic...")

# Processing logic (Either from custom input or suggestion)
final_query = None
if query:
    final_query = query
elif 'selected_prompt' in locals():
    final_query = selected_prompt

if final_query:
    # 1. Add User Message
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)
    
    # 2. Call API & Get Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing Voyage Data..."):
            try:
                # Attempt API call
                try:
                    res = requests.post(f"{API_URL}/ask", json={"question": final_query}, timeout=15)
                    if res.status_code == 200:
                        data = res.json()
                        ans = data.get("answer")
                        chart_path = data.get("chart")
                    else:
                        raise ConnectionError("API Status not 200")
                except Exception:
                    # Fallback: Call agent logic directly (Perfect for Streamlit Cloud)
                    from agent import answer_question
                    result = answer_question(final_query)
                    ans = result.get("answer")
                    chart_path = result.get("chart")
                
                st.markdown(ans)
                if chart_path and os.path.exists(chart_path):
                    st.image(chart_path)
                
                # Store to history
                st.session_state.messages.append({"role": "assistant", "content": ans, "chart": chart_path})
            except Exception as e:
                st.error(f"Engine Error: {str(e)}")

# Empty State Help
if not st.session_state.messages:
    st.markdown("""
    #### Welcome, Historian!
    Start typing in the box at the bottom of the screen or click one of the suggestion buttons above to explore the data.
    """)
