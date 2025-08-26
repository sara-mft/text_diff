#!/usr/bin/env python3
"""
build_benchmark_dashboard.py

Generates a single-file interactive HTML dashboard (dashboard.html)
from Excel benchmark result files found under ./results/.

Filename convention expected:
  <task_name>-<dataset_version>-prompt_v.<prompt_version>_<results_type>.xlsx

results_type ∈ { costs, latency, perfs, robustness }

Usage:
  python build_benchmark_dashboard.py

Output:
  dashboard.html (self-contained, references external CDNs for JS/CSS)

Dependencies:
  pip install pandas openpyxl
"""
from __future__ import annotations

import os
import re
import glob
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from html import escape

# --- Config -----------------------------------------------------------------
BASE_DIR = os.environ.get("BENCHMARK_RESULTS_BASE", "results")
OUT_HTML = "dashboard.html"
FILENAME_RE = re.compile(
    r"^(?P<task_name>.+)-(?P<dataset_version>[^-]+)-prompt_v\.(?P<prompt_version>[^_]+)_(?P<results_type>costs|latency|perfs|robustness)\.xlsx$",
    re.IGNORECASE,
)
VALID_RESULT_TYPES = ["perfs", "costs", "latency", "robustness"]


@dataclass
class ResultFile:
    task_type: str
    task_name: str
    dataset_version: str
    prompt_version: str
    results_type: str
    path: str
    df: Optional[pd.DataFrame] = None


# --- Scan and load ----------------------------------------------------------
def scan_results(base_dir: str = BASE_DIR) -> List[ResultFile]:
    pattern = os.path.join(base_dir, "*", "*", "*.xlsx")
    rows: List[ResultFile] = []
    for path in glob.glob(pattern):
        # expecting results/<task_type>/<task_name>/<file>.xlsx
        try:
            rel = os.path.relpath(path, base_dir)
            task_type, task_name, filename = rel.split(os.sep, 2)
        except ValueError:
            continue
        m = FILENAME_RE.match(os.path.basename(filename))
        if not m:
            continue
        d = m.groupdict()
        rf = ResultFile(
            task_type=task_type,
            task_name=task_name,
            dataset_version=d["dataset_version"],
            prompt_version=d["prompt_version"],
            results_type=d["results_type"].lower(),
            path=path,
        )
        rows.append(rf)
    return rows


def load_result_dataframe(rf: ResultFile) -> pd.DataFrame:
    df = pd.read_excel(rf.path, engine="openpyxl")
    # Normalize: ensure Model column
    if df.columns.size > 0 and df.columns[0].lower() != "model":
        df = df.rename(columns={df.columns[0]: "Model"})
    else:
        df.rename(columns={df.columns[0]: "Model"}, inplace=True)
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    # Convert Models to strings
    df["Model"] = df["Model"].astype(str)
    rf.df = df
    return df


# --- Build JSON data structure for embedding in HTML -----------------------
def build_data_structure(result_files: List[ResultFile]) -> Dict[str, Any]:
    # Nested structure:
    # data[task_type][task_name][dataset_version][prompt_version][results_type] = {
    #    "path": path,
    #    "columns": [...],
    #    "rows": [{...}, ...]
    # }
    data: Dict[str, Any] = {}
    for rf in result_files:
        load_result_dataframe(rf)
        tt = rf.task_type
        tn = rf.task_name
        dv = rf.dataset_version
        pv = rf.prompt_version
        rt = rf.results_type
        data.setdefault(tt, {})
        data[tt].setdefault(tn, {})
        data[tt][tn].setdefault(dv, {})
        data[tt][tn][dv].setdefault(pv, {})
        # Convert dataframe to JSON-friendly format (records)
        df = rf.df.copy()
        # Ensure consistent numeric types where possible
        for c in df.columns:
            # pandas will convert to python types on to_dict
            pass
        columns = list(df.columns)
        rows = df.to_dict(orient="records")
        data[tt][tn][dv][pv][rt] = {
            "path": rf.path,
            "columns": columns,
            "rows": rows,
            "n_rows": len(rows),
            "n_cols": len(columns),
        }
    return data


# --- HTML template (simple, uses CDNs for Plotly, Bootstrap, DataTables) -----
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Benchmark Results Dashboard</title>

  <!-- Bootstrap -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"/>

  <!-- DataTables -->
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.1/css/buttons.dataTables.min.css"/>

  <style>
    body {{ padding: 1rem; background:#f7f9fb; }}
    .sidebar {{ max-width: 300px; }}
    .content-area {{ margin-left: 320px; }}
    .card { border-radius: 12px; box-shadow: 0 6px 18px rgba(20,30,50,0.06); }
    .model-badge { font-size: 0.8rem; margin-right: 6px; }
    .no-data { color: #6c757d; padding: 1rem; }
    .chart-area { min-height: 360px; }
    .heatmap-legend { font-size: 0.9rem; color: #333; }
    .nav-task { max-height: 65vh; overflow:auto; }
  </style>
</head>
<body>
<div class="container-fluid">
  <div class="row">
    <div class="col sidebar position-fixed">
      <div class="py-3">
        <h3>📊 Benchmark Dashboard</h3>
        <p class="text-muted">Navigate: <strong>task type → task → dataset → prompt → result</strong></p>
        <hr/>
        <div id="nav-root" class="nav-task"></div>
        <hr/>
        <div>
          <label class="form-label"><strong>Filter Models</strong></label>
          <input id="global-model-filter" class="form-control" placeholder="search models (case-insensitive)">
        </div>
        <div class="mt-2">
          <label class="form-label"><strong>Filter Metrics</strong></label>
          <input id="global-metric-filter" class="form-control" placeholder="search metrics (column names)">
        </div>
        <div class="mt-3">
          <button id="reset-filters" class="btn btn-sm btn-outline-secondary">Reset filters</button>
        </div>
        <hr/>
        <small class="text-muted">Generated from files under <code>{base_dir}</code></small>
      </div>
    </div>

    <div class="col content-area">
      <div id="main-area">
        <div class="p-2">
          <h4 id="selected-title">Select a dataset on the left</h4>
        </div>
        <div id="tab-area"></div>
      </div>
    </div>
  </div>
</div>

<!-- JS libs -->
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>

<!-- DataTables + Buttons -->
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.1/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.html5.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.1/js/buttons.print.min.js"></script>

<!-- Plotly -->
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>

<script>
const DATA = {data_json};

/* Helper: create navigation tree */
function buildNav(data) {{
  const root = document.getElementById('nav-root');
  root.innerHTML = '';
  const ul = document.createElement('ul');
  ul.className = 'list-unstyled';

  Object.keys(data).sort().forEach(task_type => {{
    const liType = document.createElement('li');
    const btnType = document.createElement('button');
    btnType.className = 'btn btn-sm btn-link text-start w-100';
    btnType.innerHTML = `<strong>${{task_type}}</strong>`;
    liType.appendChild(btnType);

    const subUl = document.createElement('ul');
    subUl.className = 'list-unstyled ms-2';

    Object.keys(data[task_type]).sort().forEach(task_name => {{
      const liName = document.createElement('li');
      const btnName = document.createElement('button');
      btnName.className = 'btn btn-sm btn-outline-light text-start w-100';
      btnName.textContent = task_name;
      liName.appendChild(btnName);

      const datasetUl = document.createElement('ul');
      datasetUl.className = 'list-unstyled ms-3';

      Object.keys(data[task_type][task_name]).sort().forEach(dataset_version => {{
        const liDataset = document.createElement('li');
        const bDataset = document.createElement('div');
        bDataset.className = 'small text-muted';
        bDataset.textContent = dataset_version;

        const promptUl = document.createElement('ul');
        promptUl.className = 'list-unstyled ms-2';

        Object.keys(data[task_type][task_name][dataset_version]).sort().forEach(prompt_version => {{
          const liPrompt = document.createElement('li');
          const a = document.createElement('a');
          a.href = '#';
          a.className = 'd-block small';
          a.textContent = `prompt v${{prompt_version}}`;

          a.addEventListener('click', (ev) => {{
            ev.preventDefault();
            onSelect({{ task_type, task_name, dataset_version, prompt_version }});
          }});

          liPrompt.appendChild(a);
          promptUl.appendChild(liPrompt);
        }});

        liDataset.appendChild(bDataset);
        liDataset.appendChild(promptUl);
        datasetUl.appendChild(liDataset);
      }});

      liName.appendChild(datasetUl);
      subUl.appendChild(liName);
    }});

    liType.appendChild(subUl);
    ul.appendChild(liType);
  }});

  root.appendChild(ul);
}}

/* When a dataset+prompt is selected, build tabs for available result types */
function onSelect({{task_type, task_name, dataset_version, prompt_version}}) {{
  const selTitle = `${{task_type}} / ${{task_name}} / ${{dataset_version}} / prompt v${{prompt_version}}`;
  document.getElementById('selected-title').textContent = selTitle;

  const content = DATA[task_type][task_name][dataset_version][prompt_version];
  // result types available for this selection
  const resultTypes = Object.keys(content).sort((a,b) => {{
    const order = ['perfs','costs','latency','robustness'];
    return order.indexOf(a) - order.indexOf(b);
  }});
  const tabArea = document.getElementById('tab-area');
  tabArea.innerHTML = '';

  if (resultTypes.length === 0) {{
    tabArea.innerHTML = '<div class="card p-3"><div class="no-data">No result files found for this selection.</div></div>';
    return;
  }}

  // Build tabs
  const tabId = 'tabs-' + Math.random().toString(36).slice(2,9);
  const nav = document.createElement('ul');
  nav.className = 'nav nav-tabs';
  nav.id = tabId;

  const paneContainer = document.createElement('div');
  paneContainer.className = 'tab-content mt-3';

  resultTypes.forEach((rt, idx) => {{
    const li = document.createElement('li');
    li.className = 'nav-item';
    const a = document.createElement('a');
    a.className = 'nav-link' + (idx===0 ? ' active' : '');
    a.dataset.bsToggle = 'tab';
    a.href = `#pane-${rt}`;
    a.textContent = rt.toUpperCase();
    li.appendChild(a);
    nav.appendChild(li);

    // Pane
    const pane = document.createElement('div');
    pane.className = 'tab-pane fade' + (idx===0 ? ' show active' : '');
    pane.id = `pane-${rt}`;
    pane.innerHTML = `
      <div class="card p-3 mb-3">
        <div class="d-flex justify-content-between align-items-center mb-2">
          <div>
            <h5 class="mb-0">${{rt.toUpperCase()}}</h5>
            <small class="text-muted">Rows: ${{content[rt].n_rows}} • Columns: ${{content[rt].n_cols}}</small>
          </div>
          <div>
            <input class="form-control form-control-sm d-inline-block" style="width:220px" placeholder="Filter models..." id="input-model-${{rt}}">
          </div>
        </div>

        <div class="row">
          <div class="col-lg-6">
            <div id="chart-bar-${{rt}}" class="chart-area"></div>
            <div id="chart-line-${{rt}}" class="chart-area mt-3"></div>
          </div>
          <div class="col-lg-6">
            <div id="chart-heat-${{rt}}" class="chart-area"></div>
            <div class="mt-3">
              <table id="table-${{rt}}" class="display nowrap" style="width:100%"></table>
            </div>
          </div>
        </div>
      </div>
    `;
    paneContainer.appendChild(pane);

    // After pane created, initialize content
    setTimeout(() => {{
      initResultTypePane(content[rt], rt);
    }}, 50);
  }});

  tabArea.appendChild(nav);
  tabArea.appendChild(paneContainer);
}}

/* Initialize charts and table for a single result type object */
function initResultTypePane(obj, resultsType) {{
  const columns = obj.columns;
  const rows = obj.rows;

  // Determine numeric columns (simple heuristic)
  const numericCols = columns.filter(c => c!=='Model' && rows.some(r => typeof r[c] === 'number'));

  // Default metric picks
  const metricBar = numericCols.length ? numericCols[0] : null;
  const metricLine = numericCols.length > 1 ? numericCols[1] : metricBar;
  const heatMetrics = numericCols.slice(0, Math.min(6, numericCols.length));

  const tableId = `#table-${resultsType}`;
  const dataForTable = rows.map(r => {{
    const out = {{}}; columns.forEach(c => out[c] = r[c]===undefined ? '' : r[c]); return out;
  }});

  // Initialize DataTable
  const dtColumns = columns.map(c => ({{ title: c, data: c }}));
  const table = $(tableId).DataTable({{
    destroy: true,
    data: dataForTable,
    columns: dtColumns,
    scrollX: true,
    dom: 'Bfrtip',
    buttons: ['copy', 'csv', 'excel', 'colvis'],
    pageLength: 10
  }});

  // Model filter (pane-specific)
  const inputModel = document.getElementById(`input-model-${resultsType}`);
  inputModel.addEventListener('input', () => {{
    const q = inputModel.value.trim().toLowerCase();
    if (!q) {{
      table.search('').draw();
      return;
    }}
    // Filter rows where Model contains q
    table.search(q).draw();
  }});

  // Global model filter hookup
  document.getElementById('global-model-filter').addEventListener('input', (ev) => {{
    const q = ev.target.value.trim().toLowerCase();
    table.search(q).draw();
  }});

  // Global metric filter: hide columns not matching query
  document.getElementById('global-metric-filter').addEventListener('input', (ev) => {{
    const q = ev.target.value.trim().toLowerCase();
    table.columns().every(function(index) {{
      const colTitle = this.header().innerText.toLowerCase();
      if (!q) this.visible(true);
      else this.visible(colTitle.includes(q));
    }});
  }});

  document.getElementById('reset-filters').addEventListener('click', () => {{
    document.getElementById('global-model-filter').value = '';
    document.getElementById('global-metric-filter').value = '';
    inputModel.value = '';
    table.search('').columns().visible(true);
    table.draw();
  }});

  // Build bar chart
  const barDiv = document.getElementById(`chart-bar-${resultsType}`);
  barDiv.innerHTML = '';
  if (metricBar) {{
    const x = rows.map(r => r['Model']);
    const y = rows.map(r => (typeof r[metricBar] === 'number' ? r[metricBar] : null));
    const barData = [{{
      x, y, type: 'bar', marker: {{opacity:0.85}}, hovertemplate: '%{x}<br>{metric}: %{y}<extra></extra>'.replace('{metric}', metricBar)
    }}];
    Plotly.newPlot(barDiv, barData, {{
      title: metricBar + ' (bar)',
      margin: {{t:40}},
      xaxis: {{tickangle: -30}}
    }}, {{responsive: true}});
  }} else {{
    barDiv.innerHTML = '<div class="no-data">No numeric metrics to show bar chart.</div>';
  }}

  // Line chart
  const lineDiv = document.getElementById(`chart-line-${resultsType}`);
  lineDiv.innerHTML = '';
  if (metricLine) {{
    const x = rows.map(r => r['Model']);
    const y = rows.map(r => (typeof r[metricLine] === 'number' ? r[metricLine] : null));
    const lineData = [{ x, y, type: 'scatter', mode: 'lines+markers', marker:{size:6} }];
    Plotly.newPlot(lineDiv, lineData, {{
      title: metricLine + ' (line)',
      margin: {{t:40}},
      xaxis: {{tickangle: -30}}
    }}, {{responsive: true}});
  }} else {{
    lineDiv.innerHTML = '<div class="no-data">No numeric metrics to show line chart.</div>';
  }}

  // Heatmap
  const heatDiv = document.getElementById(`chart-heat-${resultsType}`);
  heatDiv.innerHTML = '';
  if (heatMetrics.length) {{
    const z = rows.map(r => heatMetrics.map(m => (typeof r[m] === 'number' ? r[m] : NaN)));
    const x = heatMetrics;
    const y = rows.map(r => r['Model']);
    const hm = [{ z, x, y, type:'heatmap', colorscale:'Viridis', hovertemplate: '%{y}<br>%{x}: %{z}<extra></extra>' }];
    Plotly.newPlot(heatDiv, hm, {{
      title: 'Heatmap (' + heatMetrics.join(', ') + ')',
      margin: {{t:40, l:120}}
    }}, {{responsive: true}});
  }} else {{
    heatDiv.innerHTML = '<div class="no-data">No numeric metrics to construct heatmap.</div>';
  }}
}}

/* Initialize page */
document.addEventListener('DOMContentLoaded', function() {{
  buildNav(DATA);
}});
</script>
</body>
</html>
"""

# --- Main -------------------------------------------------------------------
def main():
    rf_list = scan_results(BASE_DIR)
    if not rf_list:
        print(f"No result files found under '{BASE_DIR}'. Expected structure: {BASE_DIR}/<task_type>/<task_name>/<files>.xlsx")
        return

    data = build_data_structure(rf_list)
    # Dump JSON (ensure ascii-safe)
    data_json = json.dumps(data, default=str)

    html = HTML_TEMPLATE.format(data_json=data_json, base_dir=escape(BASE_DIR))
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard generated: {OUT_HTML} (open in your browser)")

if __name__ == "__main__":
    main()
