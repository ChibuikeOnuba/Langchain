import streamlit as st
import json
from medical_note_analyzer import analyze_clinical_note
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def create_heatmap(medications_data):
    """Create a heatmap showing medication impacts"""
    medications = [med['medication_name'] for med in medications_data]
    impact_types = ['Interactions', 'Lifestyle', 'Monitoring']
    
    # Count the number of items in each category
    values = []
    for med in medications_data:
        values.append([
            len(med['potential_interactions']),
            len(med['lifestyle_impacts']),
            len(med['monitoring_needs'])
        ])
    
    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=impact_types,
        y=medications,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Medication Impact Analysis',
        xaxis_title='Impact Categories',
        yaxis_title='Medications'
    )
    
    return fig

def create_medication_tables(medications_data):
    """Create detailed tables for each medication"""
    tables = []
    for med in medications_data:
        df = pd.DataFrame({
            'Category': ['Interactions'] * len(med['potential_interactions']) +
                       ['Lifestyle Impacts'] * len(med['lifestyle_impacts']) +
                       ['Monitoring Needs'] * len(med['monitoring_needs']),
            'Details': med['potential_interactions'] +
                      med['lifestyle_impacts'] +
                      med['monitoring_needs']
        })
        tables.append((med['medication_name'], df))
    return tables

def main():
    st.set_page_config(page_title="Clinical Note Analyzer", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🏥 Clinical Note Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This tool analyzes clinical notes to provide:
        - Medication analysis
        - Interaction warnings
        - Lifestyle recommendations
        - Monitoring requirements
        """)
        
        st.header("Settings")
        show_raw_json = st.checkbox("Show Raw JSON Output", False)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Clinical Note")
        # Add sample note button
        if st.button("Load Sample Note"):
            sample_note = """
            Patient Name: [Redacted]
            Date: December 15, 2024
            
            Current Medications:
            - Metformin 1000mg twice daily
            - Lisinopril 20mg daily
            - Atorvastatin 40mg at bedtime
            
            Social History:
            Patient works as a software developer, primarily sedentary job.
            Exercises 2 times per week with 30-minute walks.
            Diet consists of mostly prepared meals due to busy schedule.
            """
            st.session_state['clinical_note'] = sample_note
        
        clinical_note = st.text_area(
            "Clinical Note",
            value=st.session_state.get('clinical_note', ''),
            height=300,
            help="Paste the clinical note here for analysis"
        )
    
    with col2:
        st.subheader("⚙️ Analysis Options")
        include_timeline = st.checkbox("Generate Timeline", True)
        include_risks = st.checkbox("Include Risk Assessment", True)
    
    # Analysis button
    if st.button("🔍 Analyze Note", type="primary"):
        if not clinical_note.strip():
            st.error("Please enter a clinical note to analyze.")
            return
        
        with st.spinner("Analyzing clinical note..."):
            try:
                # Perform analysis
                analysis_result = analyze_clinical_note(clinical_note)
                
                # Display results
                st.markdown("---")
                st.header("📊 Analysis Results")
                
                # Medications Overview
                st.subheader("Medication Overview")
                fig = create_heatmap(analysis_result['medications'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Medication Analysis
                st.subheader("Detailed Medication Analysis")
                tabs = st.tabs([med['medication_name'] for med in analysis_result['medications']])
                
                tables = create_medication_tables(analysis_result['medications'])
                for tab, (med_name, df) in zip(tabs, tables):
                    with tab:
                        st.dataframe(df, use_container_width=True)
                
                # Lifestyle Impacts
                st.subheader("🎯 Lifestyle Recommendations")
                lifestyle = analysis_result['lifestyle_impacts']
                
                cols = st.columns(3)
                with cols[0]:
                    st.write("Daily Routine Changes")
                    for change in lifestyle['daily_routine_changes']:
                        st.markdown(f"• {change}")
                
                with cols[1]:
                    st.write("Dietary Restrictions")
                    for restriction in lifestyle['dietary_restrictions']:
                        st.markdown(f"• {restriction}")
                
                with cols[2]:
                    st.write("Activity Modifications")
                    for modification in lifestyle['activity_modifications']:
                        st.markdown(f"• {modification}")
                
                # Raw JSON output
                if show_raw_json:
                    st.subheader("Raw Analysis Data")
                    st.json(analysis_result)
                
                # Add timestamp
                st.markdown("---")
                st.caption(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    main()