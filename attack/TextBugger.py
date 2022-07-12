"""

TextBugger
===============

(TextBugger: Generating Adversarial Text Against Real-world Applications)

"""
import argparse
import sys
sys.path.append("..")

from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxModificationRate,
    InputColumnModification,
    MinWordLength,
)
from textattack.constraints.overlap import MaxWordsPerturbed, LevenshteinEditDistance
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder, BERT
from textattack.constraints.semantics import WordEmbeddingDistance

from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch

from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
    WordSwapMaskedLM,
    WordSwapWordNet,
    WordSwapQWERTY,
    WordSwapChangeLocation,
    WordSwapChangeName,
    WordSwapChangeNumber,
    WordSwapContract,
    WordSwapExtend,
)

from textattack.attack_recipes import AttackRecipe
from model.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    DebertaPrefixForSequenceClassification
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BERTAttackLi2020(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate):
        transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=48)
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxWordsPerturbed(max_percent=0.4))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        search_method = GreedyWordSwapWIR(wir_method="unk")

        return Attack(goal_function, constraints, transformation, search_method)


class TextBuggerLi2018(AttackRecipe):

    @staticmethod
    def build(model_wrapper, max_rate):
        transformation = CompositeTransformation(
            [
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                WordSwapHomoglyphSwap(),
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=0.8))
        constraints.append(MaxModificationRate(max_rate=max_rate,min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class TextFoolerJin2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate):
        transformation = WordSwapEmbedding(max_candidates=50)
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints.append(MaxModificationRate(max_rate=max_rate,min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)


class PWWSRen2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)



class Pruthi2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, max_num_word_swaps=1):
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )
        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=max_num_word_swaps),
            RepeatModification(),
        ]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)



class CheckList2020(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate):
        transformation = CompositeTransformation(
            [
                WordSwapExtend(),
                WordSwapContract(),
                WordSwapChangeName(),
                WordSwapChangeNumber(),
                WordSwapChangeLocation(),
            ]
        )
        constraints = [RepeatModification()]
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedySearch()

        return Attack(goal_function, constraints, transformation, search_method)


class DeepWordBugGao2018(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, use_all_transformations=True):
        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    WordSwapNeighboringCharacterSwap(),
                    WordSwapRandomCharacterSubstitution(),
                    WordSwapRandomCharacterDeletion(),
                    WordSwapRandomCharacterInsertion(),
                ]
            )
        else:
            transformation = WordSwapRandomCharacterSubstitution()
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(LevenshteinEditDistance(30))
        constraints.append(MaxModificationRate(max_rate=max_rate,min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)



class BAEGarg2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate):
        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        constraints.append(MaxModificationRate(max_rate=max_rate,min_threshold=1))
        goal_function = UntargetedClassification(model_wrapper, target_max_score=0.4)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return Attack(goal_function, constraints, transformation, search_method)



class A2TYoo2021(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_rate, mlm=False):
        constraints = [RepeatModification(), StopwordModification()]
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
        # constraints.append(MaxModificationRate(max_rate=0.1, min_threshold=4))
        sent_encoder = BERT(
            model_name="stsb-distilbert-base", threshold=0.9, metric="cosine"
        )
        constraints.append(sent_encoder)

        if mlm:
            transformation = transformation = WordSwapMaskedLM(
                method="bae", max_candidates=20, min_confidence=0.0, batch_size=16
            )
        else:
            transformation = WordSwapEmbedding(max_candidates=20)
            constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
        constraints.append(MaxModificationRate(max_rate=max_rate, min_threshold=4))
        goal_function = UntargetedClassification(model_wrapper, model_batch_size=32, target_max_score=0.4)
        search_method = GreedyWordSwapWIR(wir_method="gradient")

        return Attack(goal_function, constraints, transformation, search_method)


import torch, random
import pandas as pd
from torch.nn import CrossEntropyLoss

import textattack

from textattack.models.wrappers import HuggingFaceModelWrapper


def AutoAttack(attack_method):
    if attack_method=="textbugger":
        return TextBuggerLi2018
    if attack_method=="textfooler":
        return TextFoolerJin2019
    if attack_method=="pwws":
        return PWWSRen2019
    if attack_method=="bae":
        return BAEGarg2019
    if attack_method=="deepwordbug":
        return DeepWordBugGao2018
    if attack_method=="pruthi":
        return Pruthi2019
    if attack_method=="checklist":
        return CheckList2020
    if attack_method=="bert_attack":
        return BERTAttackLi2020
    if attack_method=="a2t":
        return A2TYoo2021


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--attack_method", type=str, required=True, choices=["textbugger", "textfooler", "pwws", "bae",
                                                                      "deepwordbug", "pruthi", "checklist",
                                                                      "bert_attack", "a2t"])
    parser.add_argument("--max_rate", default=0.1, type=float)
    parser.add_argument("--model_name", default='amazon-bert-normal', type=str)

    args = parser.parse_args()
    max_rate = args.max_rate
    attack_method = args.attack_method
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained('../checkpoints/{}'.format(model_name), model_max_length=256)
    if 'prefix' in model_name:
        bert = BertPrefixForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
    elif 'prompt' in model_name:
        bert = BertPromptForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
    else:
        bert = AutoModelForSequenceClassification.from_pretrained('../checkpoints/{}'.format(model_name))
    model_wrapper = HuggingFaceModelWrapper(bert, tokenizer)
    attack_module = AutoAttack(attack_method)
    attack = attack_module.build(model_wrapper, max_rate)
    random.seed(1)
    df_db_val = pd.read_csv("../../Datasets/sentiment_data/amazon/dev.tsv", sep="\t")
    df_db_val = df_db_val.sample(500, random_state=2021)
    dataset = [i for i in zip(df_db_val.sentence, df_db_val.label)]
    dataset = textattack.datasets.Dataset(dataset)

    attack_args = AttackArgs(num_examples=500,
                             log_to_txt="adv_output/{}/{}".format(model_name, attack_method))
    attacker = Attacker(attack, dataset, attack_args)

    attack_results = attacker.attack_dataset()