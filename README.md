# ⚙️ Fine-tuning Mistral 7B using QLoRA for Video Game Reviews Classification

This repository contains code for fine-tuning the Mistral 7B language model using QLoRA (Quantization-Based Low-Rank Adaptation) for the task of classifying video game reviews. The goal is to leverage self-supervised fine-tuning to improve the model's performance on a specific downstream task, namely, classifying video game reviews based on their meaning representation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction

The code in this repository fine-tunes the Mistral 7B language model using QLoRA for the task of classifying video game reviews. The following steps are performed in this codebase:

1. **Accelerator Setup**: Utilizes the Accelerate library to configure distributed training and offloading to CPUs, preparing the environment for efficient training.

2. **Load Dataset**: The video game reviews dataset is loaded using the Hugging Face Datasets library. This dataset consists of training, validation, and test splits.

3. **Tokenization**: Tokenizes the data using a predefined prompt and prepares it for fine-tuning. The data is tokenized to make the labels and input_ids match.

4. **Base Model Inspection**: The base model's initial performance on a test input is assessed to establish a baseline for comparison.

5. **Set Up LoRA**: The LoRA (Low-Rank Adaptation) framework is set up, and QLoRA adapters are applied to specific linear layers of the model.

6. **Training**: The fine-tuning process is executed, and the model is trained on the prepared dataset with QLoRA adapters.

7. **Testing**: After training, the best-performing model checkpoint is loaded, and the model is tested on a sample input to evaluate its performance.

## Installation

To use this code, you need to install the required dependencies. You can install these dependencies using pip:

```bash
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install -q -U datasets scipy ipywidgets
pip install wandb -qU
```
## Usage

To use this code for fine-tuning Mistral 7B with QLoRA for video game reviews classification, follow these steps:

1. **Accelerator Configuration**: Configure the accelerator for distributed training and offloading to CPUs, as needed for your environment.

2. **Load Dataset**: Load the video game reviews dataset for training, validation, and testing.

3. **Tokenization**: Tokenize the dataset using predefined prompts and prepare it for fine-tuning.

4. **Base Model Inspection**: Assess the base model's initial performance on test inputs.

5. **Set Up LoRA**: Configure the LoRA framework and apply QLoRA adapters to specific linear layers of the model.

6. **Training**: Fine-tune the model using the provided training arguments, and monitor the training progress.

7. **Testing**: After training, load the best-performing model checkpoint and test it on sample inputs to evaluate its performance.

## Results

The code in this repository demonstrates the fine-tuning of Mistral 7B with QLoRA for video game reviews classification. The final model achieved improved performance on the specific downstream task of classifying video game reviews based on their meaning representation. Fine-tuning the model using QLoRA adapters allows it to understand and generate meaningful responses related to video game reviews.

## License

This code is provided under an open-source license. You can find the license information in the repository.

For any questions or inquiries, please feel free to contact the project maintainers.
```
