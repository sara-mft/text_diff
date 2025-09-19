

"""
Benchmark Results Dashboard
- Single File View
- Multi-File Comparison
- Model Rankings (with Radar Plot)
"""

import re
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
        """Parse filenames like: <task_name>-<dataset_version>-prompt_v.<promptversion>_<result_type>.xlsx"""
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
    """Charts"""

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
    """Main dashboard app"""

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
        mode = st.sidebar.radio("Mode", ["Single File View", "Multi-File Comparison", "Model Rankings"])
        return mode

    # === Single File View ===
    def render_single_file(self):
        st.sidebar.subheader("File Selection")

        task_types = list(self.structure.keys())
        if not task_types:
            st.warning("No results found!")
            return

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

        # Find file
        matching_file = None
        for file_info in files:
            if (file_info['dataset_version'] == selected_dataset and
                file_info['prompt_version'] == selected_prompt and
                file_info['result_type'] == selected_result_type):
                matching_file = file_info
                break

        if not matching_file:
            st.warning("No matching file found!")
            return

        df = self.loader.load_excel_file(matching_file['full_path'])
        if df.empty:
            st.error("Failed to load data!")
            return

        st.header(f"{selected_task_name} - {selected_result_type.capitalize()} ({selected_shot_type})")

        # Visualization
        st.subheader("Visualization")
        selected_metrics = st.multiselect("Metrics", df.columns.tolist(), default=df.columns.tolist()[:3])
        if selected_metrics:
            fig = self.visualizer.create_chart(df[selected_metrics], f"{selected_result_type} ({selected_shot_type})", chart_type)
            st.plotly_chart(fig, use_container_width=True)

        # Data table
        st.subheader("Data Table")
        st.dataframe(df, use_container_width=True)

    # === Multi-file comparison ===
    def render_comparison_mode(self):
        st.header("🔄 Multi-File Comparison")
        all_files = []
        for task_type in self.structure:
            for task_name in self.structure[task_type]:
                for shot_type in ["zero_shot", "few_shot"]:
                    for file_info in self.structure[task_type][task_name][shot_type]:
                        label = f"{task_type}/{task_name}/{shot_type} - {file_info['dataset_version']} - v{file_info['prompt_version']} - {file_info['result_type']}"
                        all_files.append((label, file_info))

        selected_labels = st.multiselect("Choose files (max 5)", [f[0] for f in all_files], max_selections=5)
        if not selected_labels:
            st.info("Select files to compare")
            return

        comparison_data = {}
        for label in selected_labels:
            file_info = next(f[1] for f in all_files if f[0] == label)
            df = self.loader.load_excel_file(file_info['full_path'])
            comparison_data[label] = df

        metric = st.selectbox("Metric to Compare", sorted(set().union(*(df.columns for df in comparison_data.values()))))
        if not metric:
            return

        fig = go.Figure()
        for label, df in comparison_data.items():
            if metric in df.columns:
                fig.add_trace(go.Bar(name=label, x=df.index, y=df[metric]))
        fig.update_layout(barmode="group", title=f"Comparison on {metric}")
        st.plotly_chart(fig, use_container_width=True)

    # === Model Rankings ===
    def render_rankings(self):
        st.header("🏆 Model Rankings by Task Type")

        task_types = list(self.structure.keys())
        if not task_types:
            st.warning("No results found!")
            return

        selected_task_type = st.selectbox("Task Type", task_types)
        shot_type = st.selectbox("Shot Type", ["zero_shot", "few_shot"])

        # Collect all dataframes in this task_type + shot_type
        dfs = []
        for task_name in self.structure[selected_task_type]:
            for file_info in self.structure[selected_task_type][task_name][shot_type]:
                df = self.loader.load_excel_file(file_info['full_path'])
                if not df.empty:
                    dfs.append(df)

        if not dfs:
            st.warning("No files available for this selection")
            return

        all_metrics = sorted(set().union(*(df.columns for df in dfs)))

        st.subheader("⚙️ Ranking Options")
        ranking_metrics = st.multiselect("Metrics for Ranking", all_metrics, default=all_metrics[:2])
        if not ranking_metrics:
            st.info("Select at least one metric for ranking")
            return

        st.subheader("📊 Radar Options")
        radar_metrics = st.multiselect("Metrics for Radar Plot", all_metrics, default=all_metrics[:2])
        if not radar_metrics:
            st.info("Select at least one metric for radar plot")
            return

        # --- Compute per-model scores ---
        raw_scores = {m: {} for m in ranking_metrics}
        for df in dfs:
            available = [m for m in ranking_metrics if m in df.columns]
            if not available:
                continue
            for model in df.index:
                for m in available:
                    raw_scores[m].setdefault(model, []).append(df.loc[model, m])

        # Aggregate: mean across tasks
        model_scores = {}
        for m, models in raw_scores.items():
            for model, vals in models.items():
                avg_val = sum(vals) / len(vals)
                model_scores.setdefault(model, {})[m] = avg_val

        if not model_scores:
            st.warning("No data found for ranking metrics")
            return

        # Normalize ranking metrics for fair comparison
        norm_scores = {}
        for m in ranking_metrics:
            vals = [model_scores[model].get(m, float("nan")) for model in model_scores]
            s = pd.Series(vals, index=model_scores.keys())
            
            if s.max() == s.min():  # avoid division by zero
                s_norm = pd.Series(1.0, index=s.index)
            else:
                if METRIC_DIRECTIONS.get(m, "maximize") == "maximize":
                    s_norm = (s - s.min()) / (s.max() - s.min())
                else:  # minimize
                    s_norm = 1 - (s - s.min()) / (s.max() - s.min())
            
            for model, v in s_norm.items():
                norm_scores.setdefault(model, []).append(v)

        ranking_data = []
        for model, metrics_dict in model_scores.items():
            row = {"Model": model}
            for m in ranking_metrics:
                row[m] = metrics_dict.get(m, float("nan"))
            row["Normalized Average Score"] = sum(norm_scores.get(model, [])) / len(ranking_metrics)
            ranking_data.append(row)

        ranking = pd.DataFrame(ranking_data).sort_values("Normalized Average Score", ascending=False)

        st.subheader("📋 Ranking Table")
        st.dataframe(ranking, use_container_width=True)

        # --- Bar chart of top models ---
        st.subheader("Top Models (Bar Chart)")
        fig = px.bar(
            ranking.head(10),
            x="Model",
            y="Normalized Average Score",
            text="Normalized Average Score"
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # --- Radar plot ---
        st.subheader("Radar Plot (Per Metric)")
        top_n = st.slider("Number of Top Models", min_value=3, max_value=15, value=5)

        normalize_radar = st.checkbox("Normalize radar metrics (0–1 scale)", value=True)

        
        # Collect per-metric averages for radar metrics
        per_metric_scores = {m: {} for m in radar_metrics}
        for df in dfs:
            available = [m for m in radar_metrics if m in df.columns]
            if not available:
                continue
            for model in df.index:
                for m in available:
                    per_metric_scores[m].setdefault(model, []).append(df.loc[model, m])

        radar_df = pd.DataFrame({
            m: {model: sum(vals) / len(vals) for model, vals in per_metric_scores[m].items()}
            for m in radar_metrics
        }).fillna(0)


        if normalize_radar:
            norm_df = pd.DataFrame(index=radar_df.index)
            for m in radar_df.columns:
                s = radar_df[m]
                if s.max() == s.min():
                    s_norm = pd.Series(1.0, index=s.index)
                else:
                    if METRIC_DIRECTIONS.get(m, "maximize") == "maximize":
                        s_norm = (s - s.min()) / (s.max() - s.min())
                    else:  # minimize
                        s_norm = 1 - (s - s.min()) / (s.max() - s.min())
                norm_df[m] = s_norm
            radar_df = norm_df
        

        if radar_df.empty:
            st.info("No data available for radar plot")
            return

        fig_radar = go.Figure()
        top_models = ranking["Model"].head(top_n).tolist()
        for model in top_models:
            if model in radar_df.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_df.loc[model].values,
                    theta=radar_df.columns,
                    fill="toself",
                    name=model
                ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            height=600,
            template="plotly_white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)


    # === Run app ===
    def run(self):
        self.render_header()
        mode = self.render_sidebar()
        if mode == "Single File View":
            self.render_single_file()
        elif mode == "Multi-File Comparison":
            self.render_comparison_mode()
        elif mode == "Model Rankings":
            self.render_rankings()


def main():
    dashboard = BenchmarkDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
