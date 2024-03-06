FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

SHELL ["/bin/bash", "--login", "-c"]

# RUN echo "export PATH=/opt/conda/bin:${PATH}" >> ~/.bashrc

# Looks like conda will activate the env ONLY if we run the whole thing with bash

# Setting up the system (as 1 layer for compactness)
RUN apt-get update && apt-get install -y --no-install-recommends wget

# Getting conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  &&\
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && rm Miniconda3-latest-Linux-x86_64.sh

RUN echo ". \"/opt/miniconda/etc/profile.d/conda.sh\"" >> ~/.bashrc
ENV PATH=/opt/miniconda/bin:${PATH}

# Now we can create the environment
# RUN conda create -n mlpipeline_segmentation python=3.8 numpy scipy pandas matplotlib
SHELL ["conda", "run", "-n", "base", "/bin/bash", "--login", "-c"]

RUN conda install -y python=3.8.5
# RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torch==1.8.2 torchvision==0.9.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
RUN pip install numpy scipy matplotlib einops
RUN pip install tqdm omegaconf==2.0.5 opencv-python-headless==4.5.1.48 scikit-learn solt tensorboard medmnist
RUN pip install scikit-image==0.18.1 segmentation-models-pytorch==0.3.3 efficientnet-pytorch==0.7.1
RUN pip install pillow click ml-collections
RUN pip install pandas==1.3.5 monai==1.0.1 natsort yacs imutils
RUN pip install nibabel GeodisTK SimpleITK

# RUN echo "conda activate mlpipeline_segmentation" >> /etc/bash.bashrc

# Copying the files for the initial env init.
COPY mlpipeline/ /opt/package/mlpipeline/
COPY setup.py /opt/package/
RUN pip install -e /opt/package

# This folder needs to be mounted
RUN mkdir /opt/workdir/
RUN mkdir /opt/inference_results/
RUN mkdir /opt/visuals/
WORKDIR /opt/workdir/
# ENTRYPOINT ["conda", "run", "-n", "mlpipeline_segmentation", "/bin/bash", "-l", "-c"]
