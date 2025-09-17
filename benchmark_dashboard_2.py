"""
Simplified Benchmark Results Dashboard with Multi-File Comparison
Streamlit app for browsing benchmark results (zero_shot & few_shot as subfolders)
"""

import re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


class BenchmarkDataLoader:
    """Load and organize benchmark Excel files"""

    def __init__(self, results_path: str = "results"):
        self.results_path = Path(results_path)
        self.data_cache = {}

    def parse_filename(self, filename: str):
        """
        Parse filenames of the form:
        <task_name>-<dataset_version>-prompt_v.<promptversion>_<result_type>.xlsx
        """
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

    def scan_directory(self):
        """Scan results directory and build navigation structure"""
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
                        structure[task_type][task_name] = {"zero_shot": [], "few_shot": []}

                        for shot_dir in ["zero_shot", "few_shot"]:
                            shot_path = task_dir / shot_dir
                            if shot_path.exists() and shot_path.is_dir():
                                for file_path in shot_path.glob("*.xlsx"):
                                    file_info = self.parse_filename(file_path.name)
                                    if file_info:
                                        file_info['full_path'] = str(file_path)
                                        file_info['shot_type'] = shot_dir
                                        structure[task_type][task_name][shot_dir].append(file_info)
        return structure

    def load_excel_file(self, file_path: str) -> pd.DataFrame:
        """Load Excel with caching"""
        if file_path not in self.data_cache:
            try:
                df = pd.read_excel(file_path, index_col=0)
                self.data_cache[file_path] = df
            except Exception as e:
                st.error(f"Error loading {file_path}: {e}")
                return pd.DataFrame()
        return self.data_cache[file_path]


class BenchmarkVisualizer:
    """Create simple charts"""

    @staticmethod
    def create_chart(df: pd.DataFrame, title: str, chart_type: str = "bar") -> go.Figure:
        if df.empty:
            return go.Figure()

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
                    textposition='auto'
                ))
            fig.update_layout(
                title=title, xaxis_title="Models", yaxis_title="Values",
                barmode='group', height=500, template='plotly_white'
            )

        elif chart_type == "heatmap":
            fig = go.Figure(data=go.Heatmap(
                z=df.values, x=metrics, y=models,
                colorscale='Viridis',
                text=df.values.round(3),
                texttemplate='%{text}', colorbar=dict(title="Value")
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
            max_vals = df.max().replace(0, 1)
            for model in models[:10]:
                values = df.loc[model] / max_vals
                fig.add_trace(go.Scatterpolar(
                    r=values, theta=metrics,
                    fill='toself', name=model
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=title, height=500, template='plotly_white'
            )

        return fig


class BenchmarkDashboard:
    """Simplified dashboard app"""

    def __init__(self):
        self.loader = BenchmarkDataLoader()
        self.visualizer = BenchmarkVisualizer()
        self.structure = self.loader.scan_directory()

    def render_header(self):
        st.markdown('<h1 class="main-header">📊 Benchmark Results</h1>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Task Types", len(self.structure))
        with col2:
            total_tasks = sum(len(tasks) for tasks in self.structure.values())
            st.metric("Total Tasks", total_tasks)
        with col3:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))

    def render_sidebar(self):
        st.sidebar.title("Navigation")

        task_types = list(self.structure.keys())
        if not task_types:
            return None, False

        selected_task_type = st.sidebar.selectbox("Task Type", task_types)
        task_names = list(self.structure[selected_task_type].keys())
        selected_task_name = st.sidebar.selectbox("Task", task_names)

        shot_types = [s for s in ["zero_shot", "few_shot"] if self.structure[selected_task_type][selected_task_name][s]]
        selected_shot_type = st.sidebar.selectbox("Shot Type", shot_types)

        files = self.structure[selected_task_type][selected_task_name][selected_shot_type]

        dataset_versions = sorted(set(f['dataset_version'] for f in files))
        prompt_versions = sorted(set(f['prompt_version'] for f in files))
        result_types = sorted(set(f['result_type'] for f in files))

        selected_dataset = st.sidebar.selectbox("Dataset Version", dataset_versions)
        selected_prompt = st.sidebar.selectbox("Prompt Version", prompt_versions)
        selected_result_type = st.sidebar.selectbox("Result Type", result_types)

        chart_type = st.sidebar.radio(
            "Chart Type", ["bar", "heatmap", "radar"],
            format_func=lambda x: {"bar": "📊 Bar", "heatmap": "🔥 Heatmap", "radar": "🎯 Radar"}[x]
        )

        comparison_mode = st.sidebar.checkbox("Enable Multi-File Comparison")

        return (selected_task_type, selected_task_name, selected_shot_type,
                selected_dataset, selected_prompt, selected_result_type, chart_type), comparison_mode

    def render_main(self, filters):
        if not filters:
            st.info("👈 Select options from the sidebar to view results")
            return

        selected_task_type, selected_task_name, shot_type, dataset, prompt, result_type, chart_type = filters

        # Find matching file
        matching_file = None
        for file_info in self.structure[selected_task_type][selected_task_name][shot_type]:
            if (file_info['dataset_version'] == dataset and
                file_info['prompt_version'] == prompt and
                file_info['result_type'] == result_type):
                matching_file = file_info
                break

        if not matching_file:
            st.warning("No matching file found!")
            return

        df = self.loader.load_excel_file(matching_file['full_path'])
        if df.empty:
            st.error("Failed to load data!")
            return

        st.header(f"{selected_task_name} - {result_type.capitalize()} ({shot_type})")

        # Visualization
        st.subheader("Visualization")
        selected_metrics = st.multiselect(
            "Metrics", df.columns.tolist(),
            default=df.columns.tolist()[:min(5, len(df.columns))]
        )
        if selected_metrics:
            fig = self.visualizer.create_chart(df[selected_metrics], f"{result_type} ({shot_type})", chart_type)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Select at least one metric")

        # Data table
        st.subheader("Data Table")
        st.dataframe(
            df.style.highlight_max(axis=0, color='lightgreen')
              .highlight_min(axis=0, color='lightcoral')
              .format("{:.4f}"),
            use_container_width=True,
            height=400
        )

    def render_comparison_mode(self):
        st.header("🔄 Multi-File Comparison")

        # Gather all files
        all_files = []
        for task_type in self.structure:
            for task_name in self.structure[task_type]:
                for shot_type in ["zero_shot", "few_shot"]:
                    for file_info in self.structure[task_type][task_name][shot_type]:
                        label = f"{task_type}/{task_name}/{shot_type} - {file_info['dataset_version']} - v{file_info['prompt_version']} - {file_info['result_type']}"
                        all_files.append((label, file_info))

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
            file_info = next(f[1] for f in all_files if f[0] == label)
            df = self.loader.load_excel_file(file_info['full_path'])
            comparison_data[label] = df

        # Choose comparison type
        comparison_type = st.radio(
            "Comparison Type",
            ["Same Metric Across Files", "Statistical Summary"]
        )

        if comparison_type == "Same Metric Across Files":
            all_metrics = set()
            for df in comparison_data.values():
                all_metrics.update(df.columns)

            selected_metric = st.selectbox("Metric to Compare", sorted(all_metrics))
            if selected_metric:
                fig = go.Figure()
                for label, df in comparison_data.items():
                    if selected_metric in df.columns:
                        fig.add_trace(go.Bar(
                            name=label,
                            x=df.index,
                            y=df[selected_metric],
                            text=df[selected_metric].round(3),
                            textposition='auto'
                        ))
                fig.update_layout(
                    title=f"Comparison: {selected_metric}",
                    xaxis_title="Models", yaxis_title=selected_metric,
                    barmode='group', height=600, template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

        elif comparison_type == "Statistical Summary":
            summary_data = []
            for label, df in comparison_data.items():
                summary_data.append({
                    'File': label,
                    'Models': len(df),
                    'Metrics': len(df.columns),
                    'Mean': df.values.mean(),
                    'Std': df.values.std(),
                    'Min': df.values.min(),
                    'Max': df.values.max()
                })
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)


def main():
    dashboard = BenchmarkDashboard()
    dashboard.render_header()
    filters, comparison_mode = dashboard.render_sidebar()
    if comparison_mode:
        dashboard.render_comparison_mode()
    else:
        dashboard.render_main(filters)


if __name__ == "__main__":
    main()
