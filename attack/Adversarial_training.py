import os
import sys
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import textattack
import transformers
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models import BertForSequenceClassification, BertPrefixForSequenceClassification, \
    BertPromptForSequenceClassification
import argparse
from model.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    DebertaPrefixForSequenceClassification
)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", default='normal', type=str, choices=["normal", "prefix", "prompt"])
    # parser.add_argument("--attack_method", type=str, required=True, choices=["textbugger", "textfooler", "pwws"])

    # args = parser.parse_args()
    # model_name = 'amazon-bert-' + args.model_name

    tokenizer = AutoTokenizer.from_pretrained('bert-large-cased', model_max_length=256)
    model = BertPrefixForSequenceClassification.from_pretrained("bert-large-cased")
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    # We only use DeepWordBugGao2018 to demonstration purposes.
    attack = textattack.attack_recipes.A2TYoo2021.build(model_wrapper)

    df_train = pd.read_csv("../../Datasets/sentiment_data/amazon/train.tsv", sep="\t")
    raw_train, raw_validate = np.split(df_train.sample(6250, random_state=42), [5000,])
    # df_train = df_train.sample(50000, random_state=2020)
    sentences_train = raw_train.sentence.tolist()
    labels_train = raw_train.label.values.tolist()
    sentences_val = raw_validate.sentence.tolist()
    labels_val = raw_validate.label.values.tolist()

    train_dataset = [i for i in zip(sentences_train, labels_train)]
    train_dataset = textattack.datasets.Dataset(train_dataset)

    eval_dataset = [i for i in zip(sentences_val, labels_val)]
    eval_dataset = textattack.datasets.Dataset(eval_dataset)

    # Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5,
    # and effective batch size of 32 (8x4).

    training_args = textattack.TrainingArgs(
        num_epochs=3,
        num_clean_epochs=1,
        num_train_adv_examples=1000,
        learning_rate=5e-5,
        per_device_train_batch_size=28,
        gradient_accumulation_steps=4,
        log_to_tb=True,
    )

    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        train_dataset,
        eval_dataset,
        training_args
    )
    trainer.train()
