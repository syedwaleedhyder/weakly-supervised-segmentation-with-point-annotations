# Weakly Supervised Segmentation with Point Annotations using Partial Cross-Entropy Loss

This project utilizes the Massachusetts Buildings Dataset to train a machine learning model for building detection under weak supervision. Below are the instructions to set up and run the project.

## Prerequisites

- Python 3.x
- pip

## Installation

1. **Download the Dataset**
   
   Download the dataset from Kaggle:

   - [Massachusetts Buildings Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset)

   After downloading, extract the dataset into the directory outside the project folder i.e. ../Massachusetts Buildings Dataset

3. **Set Up the Python Environment**
   
   It's recommended to use a virtual environment to avoid conflicts with other packages:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Modify the configuration settings in `config.py` as per your experiment needs:

```python
class Config:
    batch_size = 4
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = "../Massachusetts Buildings Dataset/png/sliced"
    is_weakly_supervised = True
    point_label_percentage = 0.10  # For weak supervision
    use_wandb = True  # Set to True to enable wandb logging
```

## Running the Project

To train the model, run the following command:

```bash
python train.py
```

This script will start the training process based on the configurations set in `config.py`. Ensure that the dataset path and other settings are correctly configured.

## Monitoring

If `use_wandb` is set to `True` in the config file, ensure you have an account on [Weights & Biases](https://www.wandb.ai/) and are logged in through the CLI. This will allow you to monitor the training progress remotely.