"""
This code is adapted from the source code used in the paper
'Style Change Detection Using BERT (2020)'

Title: Style-Change-Detection-Using-BERT
Authors: Aarish Iyer and Soroush Vosoughi
Date: Jul 18, 2020
Availability: https://github.com/aarish407/Style-Change-Detection-Using-BERT
"""

import random
import re
import pickle
import time
from tqdm import tqdm
import numpy as np
import os

from transformers import BertTokenizer, BertModel
import torch

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHABET = "([A-Za-z])"
PREF = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
SUFF = "(Inc|Ltd|Jr|Sr|Co)"
STARTERS = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
WEBSITES = "[.](com|net|org|io|gov|me|edu)"
DIGITS = "([0-9])"


def paragraph_to_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(PREF, "\\1<prd>", text)
    text = re.sub(WEBSITES, "<prd>\\1", text)
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + ALPHABET + "[.] ", " \\1<prd> ", text)
    text = re.sub(ACRONYMS + " " + STARTERS, "\\1<stop> \\2", text)
    text = re.sub(ALPHABET + "[.]" + ALPHABET + "[.]" + ALPHABET + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
    text = re.sub(ALPHABET + "[.]" + ALPHABET + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + SUFF + "[.] " + STARTERS, " \\1<stop> \\2", text)
    text = re.sub(" " + SUFF + "[.]", " \\1<prd>", text)
    text = re.sub(" " + ALPHABET + "[.]", " \\1<prd>", text)
    text = re.sub(DIGITS + "[.]" + DIGITS, "\\1<prd>\\2", text)
    if "e.g." in text:
        text = text.replace("e.g.", "e<prd>g<prd>")
    if "..." in text:
        text = text.replace("...", "<prd><prd><prd>")
    if "i.e." in text:
        text = text.replace("i.e.", "i<prd>e<prd>")
    if "”" in text:
        text = text.replace(".”", "”.")
    if "\"" in text:
        text = text.replace(".\"", "\".")
    if "!" in text:
        text = text.replace("!\"", "\"!")
    if "?" in text:
        text = text.replace("?\"", "\"?")
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences


def embed_sentence(sentence, tokenizer, model):
    # Tokenize input
    sentence = tokenizer.tokenize("[CLS] " + sentence + " [SEP]")

    if len(sentence) > 512:
        sentence = sentence[:512]

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(sentence)
    # In our case we only have one sentence, i.e. one segment id
    segment_ids = [0] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    token_tensor = torch.tensor([indexed_tokens]).to(device)
    segment_tensor = torch.tensor([segment_ids]).to(device)

    with torch.no_grad():
        # Output state of last 4 layers
        output = model(token_tensor, segment_tensor, output_hidden_states=True)["hidden_states"][-4:]
        token_embeddings = torch.stack(output, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = torch.sum(token_embeddings, dim=0)
        sentence_embedding_sum = torch.sum(token_embeddings, dim=0)

    return sentence_embedding_sum


def generate_embeddings(documents):

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased').to(device)
    model.eval()

    embeddings_per_document = []
    embeddings_per_paragraph = []

    with tqdm(documents, unit="document", desc=f"Generating embeddings") as pbar:
        for doc in pbar:

            doc_embedding = torch.zeros(768)
            par_embeddings = []
            sentence_count = 0

            for par in doc:

                par_embedding = torch.zeros(768)
                sentences = paragraph_to_sentences(par)

                for sent in sentences:
                    sentence_count += 1
                    sent_embedding = embed_sentence(sent, tokenizer, model)
                    par_embedding.add_(sent_embedding)

                doc_embedding.add_(par_embedding)
                par_embeddings.append(par_embedding)

            embeddings_per_document.append(doc_embedding / sentence_count)
            embeddings_per_paragraph.append(par_embeddings)

    # Convert lists to numpy arrays
    embeddings_per_document = np.stack(embeddings_per_document)

    for i in range(len(embeddings_per_paragraph)):
        embeddings_per_paragraph[i] = np.stack(embeddings_per_paragraph[i])

    return embeddings_per_document, embeddings_per_paragraph


def main():
    from utilities import load_documents

    # Load documents
    train_docs, train_doc_ids = load_documents('train')
    val_docs, val_doc_ids = load_documents('val')

    # NB! Generating embeddings takes a long time
    train_doc_emb, train_par_emb = generate_embeddings(train_docs)
    val_doc_emb, val_par_emb = generate_embeddings(val_docs)

    # Save results
    timestring = time.strftime("%Y%m%d-%H%M")

    if not os.path.exists('./features'):
        os.makedirs('./features')

    with open('./features/' + timestring + '_doc_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/' + timestring + '_par_emb_train.pickle', 'wb') as handle:
        pickle.dump(train_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/' + timestring + '_doc_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_doc_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./features/' + timestring + '_par_emb_val.pickle', 'wb') as handle:
        pickle.dump(val_par_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
