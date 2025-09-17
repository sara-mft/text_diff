"""
Advanced Benchmark Results Dashboard
A modern, interactive web application for visualizing benchmark results
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Benchmark Results Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

class BenchmarkDataLoader:
    """Handles loading and parsing of benchmark result files"""
    
    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.data_cache = {}
        
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Parse filename to extract metadata"""
        pattern = r'^(.+?)-(.+?)-prompt_v\.(.+?)_(costs|latency|perfs|robustness)\.xlsx$'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'task_name': match.group(1),
                'dataset_version': match.group(2),
                'prompt_version': match.group(3),
                'result_type': match.group(4),
                'filename': filename
            }
        return None
    
    def scan_directory(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Scan the results directory and organize files by task type and task name"""
        structure = {}
        
        if not self.results_path.exists():
            st.error(f"Results directory '{self.results_path}' not found!")
            return structure
        
        for task_type_dir in self.results_path.iterdir():
            if task_type_dir.is_dir():
                task_type = task_type_dir.name
                structure[task_type] = {}
                
                for task_dir in task_type_dir.iterdir():
                    if task_dir.is_dir():
                        task_name = task_dir.name
                        structure[task_type][task_name] = []
                        
                        for file_path in task_dir.glob("*.xlsx"):
                            file_info = self.parse_filename(file_path.name)
                            if file_info:
                                file_info['full_path'] = str(file_path)
                                structure[task_type][task_name].append(file_info)
        
        return structure
    
    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load an Excel file with caching"""
        if file_path not in self.data_cache:
            try:
                df = pd.read_excel(file_path, index_col=0)
                self.data_cache[file_path] = df
            except Exception as e:
                st.error(f"Error loading {file_path}: {str(e)}")
                return pd.DataFrame()
        return self.data_cache[file_path]

class BenchmarkVisualizer:
    """Handles visualization of benchmark results"""
    
    @staticmethod
    def create_comparison_chart(df: pd.DataFrame, title: str, chart_type: str = "bar") -> go.Figure:
        """Create an interactive comparison chart"""
        if df.empty:
            return go.Figure()
        
        # Prepare data for visualization
        models = df.index.tolist()
        metrics = df.columns.tolist()
        
        if chart_type == "bar":
            fig = go.Figure()
            colors = px.colors.qualitative.Set3[:len(metrics)]
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Bar(
                    name=metric,
                    x=models,
                    y=df[metric],
                    marker_color=colors[i % len(colors)],
                    text=df[metric].round(3),
                    textposition='auto',
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Models",
                yaxis_title="Values",
                barmode='group',
                height=500,
                hovermode='x unified',
                showlegend=True,
                template='plotly_white'
            )
            
        elif chart_type == "heatmap":
            fig = go.Figure(data=go.Heatmap(
                z=df.values,
                x=metrics,
                y=models,
                colorscale='Viridis',
                text=df.values.round(3),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Value")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Metrics",
                yaxis_title="Models",
                height=max(400, len(models) * 40),
                template='plotly_white'
            )
            
        elif chart_type == "radar":
            fig = go.Figure()
            
            for model in models[:10]:  # Limit to 10 models for clarity
                values = df.loc[model].values
                # Normalize values to 0-1 scale for better visualization
                max_vals = df.max().values
                max_vals[max_vals == 0] = 1  # Avoid division by zero
                normalized_values = values / max_vals
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_values,
                    theta=metrics,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                title=title,
                height=500,
                template='plotly_white'
            )
        
        elif chart_type == "box":
            fig = go.Figure()
            colors = px.colors.qualitative.Set3[:len(metrics)]
            
            for i, metric in enumerate(metrics):
                fig.add_trace(go.Box(
                    y=df[metric],
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    boxmean='sd'
                ))
            
            fig.update_layout(
                title=title,
                yaxis_title="Values",
                showlegend=False,
                height=500,
                template='plotly_white'
            )
        
        return fig
    
    @staticmethod
    def create_model_ranking(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
        """Create a ranking of models based on average performance"""
        if df.empty:
            return pd.DataFrame()
        
        # Calculate average across all metrics
        avg_scores = df.mean(axis=1).sort_values(ascending=ascending)
        ranking_df = pd.DataFrame({
            'Rank': range(1, len(avg_scores) + 1),
            'Model': avg_scores.index,
            'Average Score': avg_scores.values
        })
        return ranking_df

class BenchmarkDashboard:
    """Main dashboard application"""
    
    def __init__(self):
        self.loader = BenchmarkDataLoader()
        self.visualizer = BenchmarkVisualizer()
        self.structure = self.loader.scan_directory()
        
        # Initialize session state
        if 'selected_files' not in st.session_state:
            st.session_state.selected_files = []
        if 'comparison_mode' not in st.session_state:
            st.session_state.comparison_mode = False
    
    def render_header(self):
        """Render the dashboard header"""
        st.markdown('<h1 class="main-header">🚀 Benchmark Results Dashboard</h1>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Task Types", len(self.structure))
        with col2:
            total_tasks = sum(len(tasks) for tasks in self.structure.values())
            st.metric("Total Tasks", total_tasks)
        with col3:
            total_files = sum(
                len(files) 
                for task_type in self.structure.values() 
                for files in task_type.values()
            )
            st.metric("Result Files", total_files)
        with col4:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    
    def render_sidebar(self):
        """Render the sidebar with filters"""
        st.sidebar.title("🔍 Navigation & Filters")
        
        # Task Type Selection
        task_types = list(self.structure.keys())
        if not task_types:
            st.sidebar.warning("No benchmark results found!")
            return None, None, None, None, None
        
        selected_task_type = st.sidebar.selectbox(
            "Select Task Type",
            task_types,
            help="Choose the category of benchmarks"
        )
        
        # Task Name Selection
        task_names = list(self.structure[selected_task_type].keys())
        selected_task_name = st.sidebar.selectbox(
            "Select Task",
            task_names,
            help="Choose the specific benchmark task"
        )
        
        # Get available files for selected task
        files = self.structure[selected_task_type][selected_task_name]
        
        # Extract unique values for filters
        dataset_versions = sorted(set(f['dataset_version'] for f in files))
        prompt_versions = sorted(set(f['prompt_version'] for f in files))
        result_types = sorted(set(f['result_type'] for f in files))
        
        # Dataset Version Filter
        selected_dataset = st.sidebar.selectbox(
            "Dataset Version",
            dataset_versions,
            help="Choose the dataset version"
        )
        
        # Prompt Version Filter
        selected_prompt = st.sidebar.selectbox(
            "Prompt Version",
            prompt_versions,
            help="Choose the prompt version"
        )
        
        # Result Type Filter
        selected_result_type = st.sidebar.selectbox(
            "Result Type",
            result_types,
            help="Choose the type of results to view"
        )
        
        # Additional Options
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Display Options")
        
        chart_type = st.sidebar.radio(
            "Visualization Type",
            ["bar", "heatmap", "radar", "box"],
            format_func=lambda x: {
                "bar": "📊 Bar Chart",
                "heatmap": "🔥 Heatmap",
                "radar": "🎯 Radar Chart",
                "box": "📦 Box Plot"
            }[x]
        )
        
        show_ranking = st.sidebar.checkbox("Show Model Rankings", value=True)
        show_statistics = st.sidebar.checkbox("Show Statistics", value=True)
        
        # Comparison Mode
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Comparison Mode")
        comparison_mode = st.sidebar.checkbox(
            "Enable Multi-file Comparison",
            help="Compare results across different versions or types"
        )
        
        return (selected_task_type, selected_task_name, selected_dataset, 
                selected_prompt, selected_result_type), chart_type, show_ranking, show_statistics, comparison_mode
    
    def render_main_content(self, filters, chart_type, show_ranking, show_statistics):
        """Render the main content area"""
        if not filters[0]:
            st.info("👈 Please select options from the sidebar to view results")
            return
        
        selected_task_type, selected_task_name, selected_dataset, selected_prompt, selected_result_type = filters
        
        # Find the matching file
        matching_file = None
        for file_info in self.structure[selected_task_type][selected_task_name]:
            if (file_info['dataset_version'] == selected_dataset and
                file_info['prompt_version'] == selected_prompt and
                file_info['result_type'] == selected_result_type):
                matching_file = file_info
                break
        
        if not matching_file:
            st.warning("No matching file found for the selected filters!")
            return
        
        # Load the data
        df = self.loader.load_excel_file(matching_file['full_path'])
        
        if df.empty:
            st.error("Failed to load data or file is empty!")
            return
        
        # Display title and metadata
        st.header(f"📈 {selected_task_name} - {selected_result_type.capitalize()} Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Dataset:** {selected_dataset}")
        with col2:
            st.info(f"**Prompt Version:** {selected_prompt}")
        with col3:
            st.info(f"**Models:** {len(df)}")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Visualization", "📋 Data Table", "📈 Analytics", "🔍 Model Search"])
        
        with tab1:
            # Visualization
            st.subheader("Interactive Visualization")
            
            # Metric filter
            selected_metrics = st.multiselect(
                "Select Metrics to Display",
                df.columns.tolist(),
                default=df.columns.tolist()[:min(5, len(df.columns))]
            )
            
            if selected_metrics:
                filtered_df = df[selected_metrics]
                fig = self.visualizer.create_comparison_chart(
                    filtered_df, 
                    f"{selected_task_name} - {selected_result_type.capitalize()}",
                    chart_type
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least one metric to visualize!")
        
        with tab2:
            # Data Table
            st.subheader("Raw Data Table")
            
            # Add search functionality
            search_term = st.text_input("🔍 Search models...", "")
            
            if search_term:
                filtered_models = df.index[df.index.str.contains(search_term, case=False)]
                display_df = df.loc[filtered_models]
            else:
                display_df = df
            
            # Display the dataframe with formatting
            st.dataframe(
                display_df.style.highlight_max(axis=0, color='lightgreen')
                                .highlight_min(axis=0, color='lightcoral')
                                .format("{:.4f}"),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = display_df.to_csv()
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"{selected_task_name}_{selected_result_type}_{selected_dataset}.csv",
                mime="text/csv"
            )
        
        with tab3:
            # Analytics
            st.subheader("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if show_statistics:
                    st.markdown("#### 📊 Summary Statistics")
                    st.dataframe(
                        df.describe().style.format("{:.4f}"),
                        use_container_width=True
                    )
            
            with col2:
                if show_ranking:
                    st.markdown("#### 🏆 Model Rankings")
                    
                    # Determine if lower is better based on result type
                    ascending = selected_result_type in ['costs', 'latency']
                    ranking_df = self.visualizer.create_model_ranking(df, ascending)
                    
                    st.dataframe(
                        ranking_df.style.background_gradient(
                            subset=['Average Score'],
                            cmap='RdYlGn_r' if ascending else 'RdYlGn'
                        ).format({'Average Score': '{:.4f}'}),
                        use_container_width=True,
                        hide_index=True
                    )
            
            # Correlation Analysis
            if len(df.columns) > 1:
                st.markdown("#### 🔗 Metric Correlation Matrix")
                corr_matrix = df.corr()
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_matrix.values.round(2),
                    texttemplate='%{text}',
                    colorbar=dict(title="Correlation")
                ))
                
                fig_corr.update_layout(
                    title="Metric Correlation Heatmap",
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            # Model Search and Comparison
            st.subheader("Model Comparison Tool")
            
            selected_models = st.multiselect(
                "Select models to compare",
                df.index.tolist(),
                default=df.index.tolist()[:min(3, len(df))]
            )
            
            if selected_models:
                comparison_df = df.loc[selected_models]
                
                # Create comparison visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart comparison
                    fig_comp = self.visualizer.create_comparison_chart(
                        comparison_df,
                        "Selected Models Comparison",
                        "bar"
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                
                with col2:
                    # Radar chart for multi-metric comparison
                    if len(comparison_df.columns) > 2:
                        fig_radar = self.visualizer.create_comparison_chart(
                            comparison_df,
                            "Multi-Metric Profile",
                            "radar"
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)
                    else:
                        st.info("Radar chart requires at least 3 metrics")
                
                # Detailed comparison table
                st.markdown("#### Detailed Comparison")
                st.dataframe(
                    comparison_df.T.style.background_gradient(axis=1, cmap='coolwarm')
                                        .format("{:.4f}"),
                    use_container_width=True
                )
            else:
                st.info("Select models from the dropdown above to compare them")
    
    def render_comparison_mode(self):
        """Render multi-file comparison mode"""
        st.header("🔄 Multi-File Comparison Mode")
        
        # File selection
        st.subheader("Select Files to Compare")
        
        # Create a flat list of all files with labels
        all_files = []
        for task_type in self.structure:
            for task_name in self.structure[task_type]:
                for file_info in self.structure[task_type][task_name]:
                    label = f"{task_type}/{task_name} - {file_info['dataset_version']} - v{file_info['prompt_version']} - {file_info['result_type']}"
                    all_files.append((label, file_info['full_path'], file_info))
        
        if not all_files:
            st.warning("No files available for comparison!")
            return
        
        # Multi-select for files
        selected_labels = st.multiselect(
            "Choose files to compare (max 5)",
            [f[0] for f in all_files],
            max_selections=5
        )
        
        if not selected_labels:
            st.info("Select files from the dropdown above to start comparing")
            return
        
        # Load selected files
        comparison_data = {}
        for label in selected_labels:
            file_path = next(f[1] for f in all_files if f[0] == label)
            file_info = next(f[2] for f in all_files if f[0] == label)
            df = self.loader.load_excel_file(file_path)
            comparison_data[label] = {
                'data': df,
                'info': file_info
            }
        
        # Comparison options
        comparison_type = st.radio(
            "Comparison Type",
            ["Same Metric Across Files", "Model Performance Across Files", "Statistical Summary"]
        )
        
        if comparison_type == "Same Metric Across Files":
            # Get common metrics across all files
            all_metrics = set()
            for data in comparison_data.values():
                all_metrics.update(data['data'].columns)
            
            selected_metric = st.selectbox("Select Metric to Compare", sorted(all_metrics))
            
            if selected_metric:
                fig = go.Figure()
                
                for label, data in comparison_data.items():
                    if selected_metric in data['data'].columns:
                        short_label = data['info']['result_type'] + " v" + data['info']['prompt_version']
                        fig.add_trace(go.Bar(
                            name=short_label,
                            x=data['data'].index,
                            y=data['data'][selected_metric],
                            text=data['data'][selected_metric].round(3),
                            textposition='auto'
                        ))
                
                fig.update_layout(
                    title=f"Comparison: {selected_metric}",
                    xaxis_title="Models",
                    yaxis_title=selected_metric,
                    barmode='group',
                    height=600,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Model Performance Across Files":
            # Get common models across all files
            all_models = set()
            for data in comparison_data.values():
                all_models.update(data['data'].index)
            
            selected_model = st.selectbox("Select Model to Compare", sorted(all_models))
            
            if selected_model:
                comparison_results = []
                
                for label, data in comparison_data.items():
                    if selected_model in data['data'].index:
                        avg_score = data['data'].loc[selected_model].mean()
                        comparison_results.append({
                            'File': data['info']['result_type'] + " v" + data['info']['prompt_version'],
                            'Average Score': avg_score,
                            'Dataset': data['info']['dataset_version']
                        })
                
                if comparison_results:
                    comp_df = pd.DataFrame(comparison_results)
                    
                    fig = px.bar(
                        comp_df,
                        x='File',
                        y='Average Score',
                        color='Dataset',
                        title=f"Model Performance: {selected_model}",
                        text='Average Score'
                    )
                    
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(height=500, template='plotly_white')
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        elif comparison_type == "Statistical Summary":
            # Create summary statistics for all files
            summary_data = []
            
            for label, data in comparison_data.items():
                df = data['data']
                summary_data.append({
                    'File': data['info']['result_type'] + " v" + data['info']['prompt_version'],
                    'Models': len(df),
                    'Metrics': len(df.columns),
                    'Mean': df.values.mean(),
                    'Std': df.values.std(),
                    'Min': df.values.min(),
                    'Max': df.values.max()
                })
            
            summary_df = pd.DataFrame(summary_data)
            
            # Display summary table
            st.dataframe(
                summary_df.style.background_gradient(subset=['Mean', 'Std'])
                                .format({
                                    'Mean': '{:.4f}',
                                    'Std': '{:.4f}',
                                    'Min': '{:.4f}',
                                    'Max': '{:.4f}'
                                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Create comparison visualization
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Mean Values', 'Standard Deviation', 'Min Values', 'Max Values')
            )
            
            fig.add_trace(
                go.Bar(x=summary_df['File'], y=summary_df['Mean'], name='Mean', marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=summary_df['File'], y=summary_df['Std'], name='Std', marker_color='lightgreen'),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=summary_df['File'], y=summary_df['Min'], name='Min', marker_color='lightcoral'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=summary_df['File'], y=summary_df['Max'], name='Max', marker_color='lightyellow'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Main application entry point"""
        self.render_header()
        
        # Get sidebar selections
        filters, chart_type, show_ranking, show_statistics, comparison_mode = self.render_sidebar()
        
        if comparison_mode:
            self.render_comparison_mode()
        else:
            self.render_main_content(filters, chart_type, show_ranking, show_statistics)
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888; padding: 1rem;'>
                <p>🚀 Benchmark Results Dashboard v1.0 | Built with Streamlit & Plotly</p>
                <p>© 2024 - Advanced Benchmark Visualization System</p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    """Main function to run the dashboard"""
    dashboard = BenchmarkDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
