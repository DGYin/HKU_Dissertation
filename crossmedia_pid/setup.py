from setuptools import setup, find_packages

setup(
    name="crossmedia-pid",
    version="0.1.0",
    description="CrossMedia Person Identification System",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "mlx>=0.15.0",
        "mlx-vlm>=0.1.0",
        "chromadb>=0.4.0",
        "onnxruntime>=1.16.0",
        "tokenizers>=0.14.0",
        "json-repair>=0.10.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "click>=8.1.0",
        "rich>=13.0.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "crossmedia-pid=main:cli",
        ],
    },
)
