import sys
from importlib.metadata import version

packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "torchvision",
    "torchinfo",
    "thop",
    "onnx",
    "onnxruntime",
    "mlflow",
    "memory-profiler",
    "psutil",
    "codecarbon",
    "fastapi",
    "uvicorn",
    "pytest",
    "httpx",
    "locust",
    "requests",
    "pyarrow",
    "joblib",
    "tqdm",
]

print("Python:", sys.version)

for package in packages:
    print(f"{package}: {version(package)}")