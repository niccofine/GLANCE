"""Single place for paths and constants of the GLANCE forecasting pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

RESULTS_DATASET = RESULTS / "dataset"
RESULTS_CV = RESULTS / "crossvalidation"
RESULTS_LOSSSEL = RESULTS / "lossselection"
RESULTS_FINAL = RESULTS / "finaltraining"
RESULTS_EVAL = RESULTS / "evaluation"
RESULTS_EXPLAIN = RESULTS / "explainability"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_POSTHOC = RESULTS / "posthoc"

GRADES_XLSX = DATA / "raw" / "DatasetComplete.xlsx"
GRADES_SHEET = "AMD"
RETCLIP_FEATURES = DATA / "raw" / "retclip_features" / "latentvec.npy"
RETCLIP_IMAGE_PATHS = DATA / "raw" / "retclip_features" / "imagepaths.npy"

EDO_DIR = ROOT.parent / "Datasets" / "AMD_study" / "Data" / "EDO"
RETCLIP_CODE = ROOT.parent / "external" / "RET-CLIP-PLUS"
RETCLIP_CKPT = ROOT.parent / "models" / "ret-clip-plus" / "checkpoints" / "ret-clip.pt"

LATE_THRESHOLD = 4.0
VISITS_PER_YEAR = 2.0

GRID_STEP_Y = 0.5
GRID_MAX_Y = 10.0
RNN_NATIVE_YEARS = (1.0, 2.0, 3.0, 5.0, 8.0)

DROP_TOL = 0.05
N_BOOT = 1000
BOOT_SEED = 12345
