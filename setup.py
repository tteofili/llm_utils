import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="llm_utils",
    version="0.0.1",
    author="Tommaso Teofili",
    author_email="tommaso.teofili@gmail.com",
    description="LLM Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url= 'https://github.com/tteofili/llm_utils.git',
    packages=['llm_utils'],
    install_requires=[
          'pandas',
          'numpy',
          'langchain',
          'transformers',
          'openai',
          'arxiv',
          'argparse',
          'jinja2'
      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: Apache Software License',
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)
