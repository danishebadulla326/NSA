# Replication Instructions for "Normalized Space Alignment: A Versatile Metric for Representation Space Analysis"

This repository contains code and instructions for replicating the results of the paper titled "Normalized Space Alignment: A Versatile Metric for Representation Space Analysis." 
The experiments are divided into four parts: **NSA - AutoEncoder**, **Adversarial Analysis**, **GNN Analysis** and **CNN Analysis**. Follow the steps below to replicate the results:

## Requirements
- cuda 11.7
- Python 3.9
- Conda

### Steps
- Create a conda environment with python 3.9
- Install the required libraries with `pip install -r requirements.txt`


## Part 1: NSA - AutoEncoder

### Step 1: Data Download
- Navigate to the `NSA/NSA_AE/` directory.
- Run the `Download Data.ipynb` notebook as is. This will download all the required datasets to the 'data' folder.

### Step 2: AutoEncoder Training
- In the same directory, run the `AE training.ipynb` notebook four times, each for a different dataset.
- Modify the `dataset_name` and `input_dim` in the config cell according to the dataset being used. Here are the dataset names and input dimensions:

    - MNIST: 28x28
    - F-MNIST: 28x28
    - CIFAR-10: 32x32x3
    - COIL-20: 128x128
    - LinkPrediction/Cora: 256
    - LinkPrediction/Citeseer: 256
    - LinkPrediction/Amazon: 256
    - LinkPrediction/Pubmed: 256

### Step 3: PCA and UMAP Embeddings
- Still in the same directory, run the `PCA and UMAP.ipynb` notebook four times, changing the `dataset_name` for each run.

### Step 4: Visualization and Metrics
- Run the `Visualization and Metrics.ipynb` notebook four times, changing the `dataset_name` accordingly.

### Step 5: Downstream analysis.ipynb
- Run this notebook to generate the Knowledge Distillation results for Link Prediction, Semantic Text Similarity and tSNE visualization.

## Part 2: Adversarial Analysis

### Step 1: Running the models
- Go to the `NSA/Adversarial Analysis` directory.
- Create 2 folders `accuracy_vals` and `feature_vals` in `notebooks folder`
- Run the `adversarial analysis.ipynb` notebook five times, each with a different perturbation rate. Change the perturbation rates to: `[0.05, 0.1, 0.15, 0.2, 0.25]`

### Step 2: Visualization
- Run the `Visualization.ipynb` notebook to generate the results

- Run the `Nodewise Adversarial Analysis.ipynb` notebook to obtain the nodewise analysis

## Part 3: GNN Analysis

### Step 1: GNN Analysis
- Navigate to the `NSA/GNN_analysis` directory.
- There are four notebooks, one for each GNN architecture. Run them in any order twice to generate two different runs. For each notebook, the `dataset_name` can be changed. Use `seed` 1234567 for the first run and `seed` 12345 for the second run.

### Step 2: Heatmap Visualization
- Run metric validation tests after completing all four notebooks twice. Change the dataset name in cell 5.

## Part 4: CNN Analysis

## Step 1: Download data and run evaluation pass on models
- Use ImageNet\_Tests.ipynb notebook to download the data. To download the validation data, we obtain the data from the Kaggle. If you plan on following the same steps, you will require a Kaggle account. You can also obtain the ImageNet dataset through your own sources. You will need to change the data loading path in the notebook for the code to work.

##Step 2: Generate Heatmaps
- Use the Generate\_Heatmaps.ipynb notebook to generate the heatmaps using CKA, NSA and RTD and save them.



These instructions will guide you through replicating the results of the paper. Please refer to the specific notebooks in the provided directory for detailed code and analysis.
