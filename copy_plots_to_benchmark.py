import os
import shutil

ARTIFACT_DIR = r"C:\Users\pshin\.gemini\antigravity-ide\brain\4a7bbe35-fd56-4870-a940-d8dff5ae0792"
BENCHMARK_PLOTS_DIR = r"c:\Users\pshin\CODEE\research\benchmark\plots"
os.makedirs(BENCHMARK_PLOTS_DIR, exist_ok=True)

files = [
    "autoresearch_pareto_frontier.png",
    "auto_research_2hr_trajectory.png",
    "internet_hypothesis_pareto.png",
    "master_project_timeline_metrics.png"
]

for fname in files:
    src = os.path.join(ARTIFACT_DIR, fname)
    dst = os.path.join(BENCHMARK_PLOTS_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {fname} to {dst}")
    elif os.path.exists(dst):
        shutil.copy(dst, src)
        print(f"Copied {fname} to {src}")
