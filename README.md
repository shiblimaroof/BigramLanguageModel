# Bigram Language Model

A simple Bigram Language Model implemented in PyTorch. This model learns character-level dependencies from a given text corpus and generates new text based on learned patterns.

Features

	•	Character-level language modeling
	•	Bigram-based token prediction
	•	Text generation using a learned distribution
	•	PyTorch implementation for efficient training

How It Works

	1.	Reads the input text and creates a vocabulary of unique characters.
	2.	Encodes text into numerical representations.
	3.	Uses an embedding-based lookup table to predict the next character given the previous one.
	4.	Trains the model using cross-entropy loss.
	5.	Generates new text by sampling characters sequentially.

Installation

Prerequisites

Make sure you have Python 3.x and PyTorch installed:

pip install torch

Clone the Repository

git clone https://github.com/shiblimaroof/BigramLanguageModel.git
cd BigramLanguageModel

Usage

Train the Model

Run the following command to train the model:

python bigram_model.py

Generate Text

Modify the script to generate text after training:

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))