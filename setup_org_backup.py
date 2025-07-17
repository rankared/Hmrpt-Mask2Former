from setuptools import setup, find_namespace_packages

setup(name='Mask2Former',
      python_requires=">=3.10",
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
      packages=find_namespace_packages(
        # Explicitly include your custom HMR-PT package and the scripts directory if it contains modules
        include=['hmr_pt', 'hmr_pt.*', 'scripts', 'scripts.*'],
        # Explicitly exclude the 'data' directory, as it's not a Python package
        exclude=['data', 'data.*', 'output', 'output.*'] # Also exclude 'output' if it's a top-level dir
      ),
      )
