"""
WizardAI SDK - setup.py
PyPI distribution configuration.
"""
from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# ✅ Read version SAFELY - only extract the __version__ line
version = "1.0.0"
init_path = Path(__file__).parent / "wizardai" / "__init__.py"
if init_path.exists():
    for line in init_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="wizardai-sdk",
    version=version,
    author="WizardAI Contributors",
    author_email="hello@wizardai.dev",
    description=(
        "A powerful, all-in-one Python SDK for AI integration – combining "
        "conversational AI, computer vision, speech I/O, and more."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VIkasPatel22/wizardai-sdk",
    project_urls={
        "Bug Tracker": "https://github.com/VIkasPatel22/wizardai-sdk/issues",
        "Documentation": "https://github.com/VIkasPatel22/wizardai-sdk#readme",
        "Source": "https://github.com/VIkasPatel22/wizardai-sdk",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Multimedia :: Video :: Capture",
    ],
    python_requires=">=3.9",
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "anthropic": ["anthropic>=0.20.0"],
        "vision": ["opencv-python>=4.7.0"],
        "speech": [
            "SpeechRecognition>=3.10.0",
            "pyttsx3>=2.90",
        ],
        "gtts": ["gtts>=2.3.0", "pygame>=2.4.0"],
        "whisper": ["openai-whisper>=20230918", "numpy>=1.24.0"],
        "all": [
            "openai>=1.0.0",
            "anthropic>=0.20.0",
            "opencv-python>=4.7.0",
            "SpeechRecognition>=3.10.0",
            "pyttsx3>=2.90",
            "gtts>=2.3.0",
            "pygame>=2.4.0",
            "openai-whisper>=20230918",
            "numpy>=1.24.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
            "ruff>=0.1.0",
            "twine>=4.0",
            "build>=0.10",
        ],
    },
    keywords=[
        "ai", "chatbot", "openai", "anthropic",
        "speech-recognition", "text-to-speech", "computer-vision",
        "opencv", "nlp", "machine-learning", "sdk", "wizardai",
    ],
    include_package_data=True,
    zip_safe=False,
)
