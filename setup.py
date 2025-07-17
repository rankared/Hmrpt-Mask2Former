from setuptools import setup, find_packages

setup(
    name='Hmrpt-Mask2Former',
    version='0.0.1', # Update this version number as your project evolves
    packages=find_packages(
        # Include your HMR-PT package and the scripts directory 
        include=['hmr_pt', 'hmr_pt.*', 'scripts', 'scripts.*'],
        # Explicitly exclude the 'data' directory, as it's not a Python package
        exclude=['data', 'data.*', 'output', 'output.*'] # Also exclude 'output' if it's a top-level dir
    ),
    # Project's dependencies. in setup.py for pip install -e.
    #install_requires=,
    install_requires=[
          "transformers",
          "datasets",
          "evaluate",
          "lightning",
          "tqdm",
          "matplotlib",
          "opencv-python",
          "scipy"
      ],
    python_requires='>=3.10', # Minimum Python version
    # author='Rajesh Ankareddy',
    # description='HMR-PT: Hierarchical Mask-Refinement Prompt Tuning for Image Segmentation',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/rankared', 
    )
