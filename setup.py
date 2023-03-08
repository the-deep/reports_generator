from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name="reports_generator",
    # url="https://github.com/the-deep/deep-experiments/tree/test-summarization-lib",
    author="Data Friendly Space",
    author_email="",
    # Needed to actually package something
    description="A summarization tool designed to extract the most relevant information from long texts.",
    packages=find_packages(where="src"),  # include all packages under src
    package_dir={"": "src"},
    include_package_data=True,
    # Needed for dependencies
    install_requires=[
        "scikit-learn>=0.24.2",
        # "torch>=1.10.2",
        "protobuf==3.19.5",
        "networkx>=2.0",
        # "nltk>=3.6.7",
        "transformers>=4.9.2",
        "torch@http://download.pytorch.org/whl/cpu/torch-1.9.1%2Bcpu-cp38-cp38-linux_x86_64.whl",
        "sentencepiece",
    ],
    # *strongly* suggested for sharing
    version="0.2",
    # The license can be anything you like
    license="Apache-2.0",
    # We will also need a readme eventually (there will be a warning)
    long_description=open("README.md").read(),
)
