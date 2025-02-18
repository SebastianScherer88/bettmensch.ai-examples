from enum import Enum
from typing import Dict, List

from setuptools import find_packages, setup


class SDKExtras(Enum):
    gpt_1: str = "gpt-1"
    annotated_transformer: str = "annotated-transformer"
    bert: str = "bert"
    test: str = "test"


def get_extra_requirements() -> Dict[str, List[str]]:
    """Produces the requirements dictionary for the sdk's extras."""

    extra_requirements = {
        SDKExtras.gpt_1.value: [
            "jaxtyping==0.2.38",
            "torch==2.3.1",  # pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 for GPU support # noqa: E501
            "transformers==4.48.3",
            "datasets==3.2.0",
            "spacy==3.8.2",
            "GPUtil==1.4.0",
            "numpy==1.24.1",
            "scipy==1.14.1",
        ],
        SDKExtras.annotated_transformer.value: [
            "torchtext==0.18.0",
            "torchdata==0.9.0",
            "portalocker==2.10.1",
            "spacy==3.8.2",
            "torch==2.3.1",  # pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 for GPU support # noqa: E501
            "GPUtil==1.4.0",
            "numpy==1.24.1",
            "scipy==1.14.1",
        ],
        SDKExtras.bert.value: [
        ],
        SDKExtras.test.value: [  # noqa: W503
            "pytest==8.2.2",
            "pytest-order==1.2.1",
        ]
    }

    return extra_requirements


setup(
    name="bettmensch_ai_examples",
    version="0.2.0",
    author="Sebastian Scherer @ Github:SebastianScherer88",
    author_email="scherersebastian@yahoo.de",
    packages=find_packages(),
    license="LICENSE.txt",
    description="A collection of example implementations for the bettmensch.ai package.",  # noqa: E501
    long_description=open("README.md").read(),
    install_requires=[
        "bettmensch.ai[pipelines] @ git+https://github.com/SebastianScherer88/bettmensch.ai.git@master#egg=bettmensch_ai&subdirectory=sdk",
    ],
    extras_require=get_extra_requirements(),
    python_requires=">=3.8.0,<3.13.0",
    include_package_data=True,
)
