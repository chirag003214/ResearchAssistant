import streamlit as st
import pandas as pd
import plotly.express as px
from src.ingestion import load_documents
from src.retriever import get_index, get_query_engine
from src.analysis import extract_data_from_text
import os
import shutil

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide")

st.title("🔬 AI Research Assistant & Analytics")
st.markdown("### RAG + Data Science Pipeline")

# --- SESSION STATE INITIALIZATION ---
# This keeps the AI "Brain" alive in memory while you use the app
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

# --- SIDEBAR: SETTINGS & UPLOAD ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            # 1. Clear old data
            if os.path.exists("data/raw"):
                shutil.rmtree("data/raw")
            os.makedirs("data/raw", exist_ok=True)
            
            # 2. Save new files
            for uploaded_file in uploaded_files:
                with open(os.path.join("data/raw", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # 3. Build the Brain (Index)
            with st.spinner("Indexing documents..."):
                docs = load_documents()
                # Store the index in Session State so it persists
                st.session_state.vector_index = get_index(docs)
                st.success("✅ Indexing Complete!")
        else:
            st.warning("Please upload a PDF first.")

# --- MAIN TAB LAYOUT ---
tab1, tab2 = st.tabs(["💬 Chat with Papers", "📊 Analytics Dashboard"])

# --- TAB 1: RAG CHATBOT ---
with tab1:
    st.header("Ask Questions")
    query = st.text_input("Ask something about the research papers:", placeholder="What is the accuracy trend over the years?")
    
    if query:
        if st.session_state.vector_index is None:
            st.warning("⚠️ Please upload and process documents first!")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Get engine from the stored index
                    engine = get_query_engine(st.session_state.vector_index)
                    response = engine.query(query)
                    
                    st.markdown(f"**🤖 AI Answer:**")
                    st.write(response.response)
                    
                    with st.expander("View Source Documents"):
                        for node in response.source_nodes:
                            st.caption(f"Page {node.metadata.get('page_label')}: {node.text[:200]}...")
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab2:
    st.header("Structured Data Extraction")
    
    if st.button("Extract Metrics from Documents"):
        with st.spinner("Analyzing text for data points..."):
            docs = load_documents()
            all_metrics = []
            
            # 1. Setup Progress Bar
            docs_to_process = docs[:] # Analyze ALL pages
            total_docs = len(docs_to_process)
            progress_bar = st.progress(0)
            
            # 2. Debug Container (See what the AI is actually thinking)
            debug_expander = st.expander("🕵️ Debug: Raw Extracted Data", expanded=False)
            
            for i, doc in enumerate(docs_to_process): 
                # Analyze text
                metrics = extract_data_from_text(doc.text[:2000]) 
                
                # Assign a default year if missing (Good fallback for RAG)
                for m in metrics:
                    if m.year is None:
                        m.year = 2013  # Default to paper year (or handle dynamically)
                
                all_metrics.extend(metrics)
                
                # Show raw extraction in debug tab
                if metrics:
                    with debug_expander:
                        st.write(f"**Page {i+1} Found:**", metrics)

                # Update Progress
                progress_value = (i + 1) / total_docs
                progress_bar.progress(min(progress_value, 1.0))
            
            progress_bar.empty()

            # 3. Display Data (REMOVED THE STRICT FILTER)
            if all_metrics:
                data = [
                    {
                        "Year": m.year, 
                        "Metric": m.metric_name, 
                        "Value": m.metric_value, 
                        "Unit": m.unit, 
                        "Paper": m.paper_title
                    } 
                    for m in all_metrics # <--- Removed 'if m.year is not None'
                ]
                df = pd.DataFrame(data)
                
                st.success(f"✅ Found {len(df)} data points!")
                st.dataframe(df)
                
                if not df.empty:
                    st.subheader("📈 Metric Trends")
                    fig = px.line(df.sort_values("Year"), x="Year", y="Value", color="Metric", symbol="Paper", title="Performance Metrics Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("📊 Comparison")
                    # Use a scatter plot or bar chart if years are identical
                    fig2 = px.bar(df, x="Metric", y="Value", color="Paper", barmode="group", title="Metric Comparison")
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("⚠️ No structured data found. Check the 'Debug' expander above to see if the AI read anything.")