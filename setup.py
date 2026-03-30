from setuptools import setup

setup(
    name="wizardenv",
    version="2.1.2",
    author="Vikas Patel",
    description="All-in-one AI SDK powered by Sagittarius Labs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://sagittarius-labs.pages.dev/",
    py_modules=["wizardai"],
    install_requires=[
        "requests>=2.28.0",
    ],
    extras_require={
        "vision": ["opencv-python>=4.7.0", "numpy>=1.24.0"],
        "speech": ["SpeechRecognition>=3.10.0", "pyttsx3>=2.90", "gtts>=2.3.0", "pygame>=2.4.0"],
        "full": [
            "opencv-python>=4.7.0", "numpy>=1.24.0",
            "SpeechRecognition>=3.10.0", "pyttsx3>=2.90", 
            "gtts>=2.3.0", "pygame>=2.4.0"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
