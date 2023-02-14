import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md") as fh:
    long_description = fh.read()

setuptools.setup(
    name='centernet',
    version="0.0.1",
    description="Object detection training and inference pipeline using CenterNet algorithm. Built on PyTorch Lightning and ONNX Runtime. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License ",
        "Programming Language :: Python :: 3.9.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence ", 
        "Topic :: Scientific/Engineering :: Computer Vision", 
    ],
    install_requires=required,
    author='Uğurcan Özalp',
    author_email='uurcann94@gmail.com'
 )