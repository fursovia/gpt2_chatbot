import torch
import numpy as np
import pandas as pd
import os
import argparse
import faiss
from torch.utils.data import DataLoader
from train import load_checkpoint, default_params
from model import Model, PretrainedModel
from tqdm import tqdm
from data_utils import CsvDataset, load_vocab, get_tokenizer, crop_or_pad
import pickle

PAD_VALUE = 123  # '¿'

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda', default='0')
parser.add_argument('-dd', '--data_dir', default='data')
parser.add_argument('-md', '--model_dir', default='experiments')
parser.add_argument('--model', default='GPT', choices=['DAN', 'GPT', 'GPT-2'])
parser.add_argument('-l', '--loss', default='triplet', choices=['triplet', 'bce'])
parser.add_argument('-e', '--epoch', default='2')
parser.add_argument('-t', '--train', default='True', choices=['True', 'False'])


class Predictor:
    def __init__(self, model, tokenizer, k=1):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.k = k
        self.vect2text = {}
        
    def fit(self, dataloader):
        idx = 0
        for (_, _), (answer, _) in tqdm(dataloader):
            #if (idx % 100):
            #    print(idx)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            answer = answer.to(device)
            vector = self.model(answer)
            self.vect2text[vector] = answer
            idx+=1
        
        vectors = torch.stack(list(self.vect2text.keys()))
        
        emb_size = vectors.shape[2]
        self.faiss_index = faiss.IndexFlat(emb_size)
        
        self.faiss_index.verbose = True
        #print(vectors.shape[0], vectors.shape[2])
        self.faiss_index.add(vectors.detach().cpu().numpy().reshape(vectors.shape[0], vectors.shape[2]))
    
            
    def predict(self, text, max_len=50):
        if (type(text) == str):
            text = [text]
            
        text_tokenize = [self.tokenizer.encode(sentence) for sentence in text]
        text_len = torch.tensor(len(text_tokenize)) if len(text_tokenize) <= max_len else torch.tensor(max_len)
        cropped_text = crop_or_pad(text_tokenize, pad_value=PAD_VALUE, max_len=max_len)
        
        predicted_answers = []
        text_new = []
        cnt = 0
        #print("text before enumerate", cropped_text)
        for idx, sentence in enumerate(cropped_text):
            #print(sentence)
            if (type(sentence)== list):
                if len(list(sentence))!=max_len:
                    cnt+=1
                    cropped_text[idx].extend([PAD_VALUE]*(max_len-len(sentence)))
                    
        cropped_text = cropped_text[:cnt]
        cropped_text = np.array(cropped_text)
        #print(text)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cropped_text = torch.tensor(cropped_text).to(device)
        #print(cropped_text)
        vectors = self.model(cropped_text)
            
        _, indexes = self.faiss_index.search(vectors.detach().cpu().numpy(), k=self.k)
        
        df = pd.read_csv(os.path.join(args.data_dir, f'cutted_test.csv'))["answer"]
        for index in indexes:
            #print(list(df[index])[0])
            predicted_answers.append(list(df[index])[0])
        #print(predicted_answers)          
        return predicted_answers
            
def prediction(current_sentence, previous_sentences):
                         
    pred_answer = pred.predict(current_sentence)
    previous_sentences.append(pred_answer)
    
    return pred_answer, previous_sentences

if __name__ == '__main__':
    
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    params = default_params()

    ntokens = sum(1 for _ in open(os.path.join(args.data_dir, 'vocab.txt'))) + 1

    if args.model == 'DAN':
        model = Model(emb_dim=64, ntokens=ntokens, hidden_dim=32, output_dim=16).to(device)
        vocab = load_vocab(os.path.join(args.data_dir, 'vocab.txt'))
        tokenizer = None
    elif args.model in ['GPT', 'GPT-2']:
        model = PretrainedModel(model_name=args.model).to(device)
        vocab = None
        tokenizer = get_tokenizer(args.model)
    else:
        raise NotImplementedError(f'{args.model} --- no such model')
        
    model = load_checkpoint(model, os.path.join(args.model_dir,f'checkpoint_{args.epoch}'))
    
    pred = Predictor(model, tokenizer)
    
    dataset = CsvDataset(csv_path=os.path.join(args.data_dir, f'data.csv'),
            vocab=vocab,
            max_len=50,
            tokenizer=tokenizer
        )
    dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=True
        )
    if args.train==True:
        pred.fit(dataloader)
        with open('pred.pickle', 'wb') as f:
            pickle.dump(pred, f)
    else:
        with open('data.pickle', 'rb') as f:
            pred = pickle.load(f)
    

    pred_answers = pred.predict('Hello. What are you thinking about work?')
    print(pred_answers)  