from setuptools import setup, find_packages

setup(
    name='SmartSummarizerRAG',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'chromadb', 'langchain', 'langchain-community==0.2.10', 'langchain_experimental', 'langchainhub',
        'llama-cpp-python', 'sentence-transformers', 'tiktoken', 'umap-learn', 'matplotlib', 'bs4',
        'numpy', 'pandas', 'scikit-learn', 'torch', 'transformers'
    ],
)
