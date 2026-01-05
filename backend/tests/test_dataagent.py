import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.data_agent import (
  build_preprocessor,
  load_pima,
  load_diabetes_prediction,
  load_mohammed,
  load_diabetes_readmission,
  DataHandlingAgent,
  FEATURES,
)
# ======================================================================================


DATASETS = {
  "pima": {
    "loader": load_pima,
    "path": Path("data/raw/concluded/pima_indians.csv"),
  },
  "diabetes_dset1": {
    "loader": load_diabetes_prediction,
    "path": Path("data/raw/concluded/diabetes_dset1.csv"),
  },
  "mohammed": {
    "loader": load_mohammed,
    "path": Path("data/raw/concluded/mohammed.csv"),
  },
  "diabetes_dset2_readmission": {
    "loader": load_diabetes_readmission,
    "path": Path("data/raw/concluded/diabetes_dset2.csv"),
  },
}


def test_dataset(name, loader, path):
  print(f"\n=== Testing dataset: {name} ===")

  # 1. Load dataset
  df = loader(path)
  print("Loaded shape:", df.shape)

  # 2. Fit preprocessor on this dataset
  pre = build_preprocessor()
  pre.fit(df[FEATURES])

  agent = DataHandlingAgent(pre)

  # 3. Batch ingest
  out = agent.process_dataset(
    df,
    include_target=False,
  )

  print("Raw shape:", out.raw.shape)
  print("Cleaned shape:", out.cleaned.shape)

  # assert out.raw.shape[0] == df.shape[0]
  assert out.views is not None
  assert out.views_model is not None
  assert "clinical" in out.views
  assert "lab" in out.views_model
  assert out.cleaned.shape[0] == df.shape[0]

  print("Clinical view head:")
  print(out.views["clinical"].head())

  print("Lab model view head:")
  print(out.views_model["lab"].head())

  print(f"✓ {name} passed")


# --------------
def main():
  print("=== Multi-dataset DataHandlingAgent test ===")

  for name, cfg in DATASETS.items():
    loader = cfg["loader"]
    path = cfg["path"]
    test_dataset(name, loader, path)
  
  print("\n✓ All datasets passed ingestion test")




# -------------------------------------
if __name__ == "__main__":
  main()
