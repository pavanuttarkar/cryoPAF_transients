from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""


def read_requirements(filename="requirements.txt"):
    """Parse a requirements file, skipping blanks, comments, and pip options."""
    path = HERE / filename
    if not path.exists():
        return []
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #")[0].strip()  # strip inline comments
        if not line or line.startswith(("#", "-")):  # skip comments, -r, -e, --index-url, etc.
            continue
        requirements.append(line)
    return requirements

setup(
    name="cryopipe",
    version="0.0.1",
    description="A cryoPAF candidate pipeline to classify and search for transients.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Pavan and Claude",
    author_email="puttarkar@mpifr-bonn.mpg.de",
    url="https://github.com/pavanuttarkar/cryoPAF_transients",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="cryoPAF, FRB, transients",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),    
    extras_require={
        "dev": [
            "pytest",
            "build",
            "twine",
        ],
    },
    entry_points={
        "console_scripts": [
            "run_cand_pipe=src.run_cand_pipe:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
