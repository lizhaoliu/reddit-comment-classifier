# Reddit Comment Sentiment Classifier

## Overview
This is a machine learning model for determining the sentiment against Reddit comments on crypto.

## Model
The sentiment classifier is fine-tuned based on a `distilbert` model. The training data and validation data are extracted from this [CSV file](https://gist.github.com/flotothemoon/e060935138f5686efae6911bce45e7b3). Train/validation split is 75/25.

The model accuracy, precision and recall on the validation set are 0.9148, 0.9333 and 0.9090.

The Jupyter notebook for fine-tuning the model is `train.ipynb`.

## Build and Run Server
1. Clone the project.
```
git clone https://github.com/lizhaoliu/reddit-comment-classifier.git && cd reddit-comment-classifier
```
2. Download the [model file](https://drive.google.com/file/d/1tijE5McKEVOwtFAzgdgxC6AhIg1vYc5Y/view?usp=sharing) and and extract everything to the `model` directory, i.e.
```
reddit-comment-classifier/
├── model
│   ├── ckpt
│   │   ├── config.json
│   │   ├── optimizer.pt
│   │   ├── pytorch_model.bin
│   │   ├── rng_state.pth
│   │   ├── scheduler.pt
│   │   ├── trainer_state.json
│   │   └── training_args.bin
...
```
3. Create a Conda environment and install Python dependencies.
```
conda create -n reddit-sentiment-classifier -y -c pytorch -c huggingface python=3.10 pytorch scikit-learn pandas transformers flask && \
conda activate reddit-sentiment-classifier
```
4. Bootstrap the Flask server, the server runs on `localhost:12345`.
```
python server.py
```
5. You can also make `POST` requests to `/predict` endpoint containing a text data.
