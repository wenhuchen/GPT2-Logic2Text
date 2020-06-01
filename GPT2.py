import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer
from DataLoader import *
from utils import sample_sequence
import torch.optim as optim
import math
import sys
import pandas
import os
import numpy
import nltk
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device('cuda')

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--do_train', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true", help="whether to compute the BLEU scores on test split")
    parser.add_argument('--epoch', default=10, type=int, help="whether to train or test the model")
    parser.add_argument('--batch_size', default=5, type=int, help="whether to train or test the model")
    parser.add_argument('--learning_rate', default=2e-6, type=float, help="whether to train or test the model")
    parser.add_argument('--dataset', default='table', type=str, help="whether to train or test the model")
    parser.add_argument('--every', default=50, type=int, help="whether to train or test the model")
    parser.add_argument('--load_from', default='', type=str, help="whether to train or test the model")
    parser.add_argument('--id', default='models', type=str, help="specify the id of the experiment")
    parser.add_argument('--max_len', default=800, type=int, help="whether to train or test the model")
    parser.add_argument('--dim', default=768, type=int, help="whether to train or test the model")
    parser.add_argument('--layers', default=3, type=int, help="whether to train or test the model")
    parser.add_argument('--head', default=4, type=int, help="whether to train or test the model")
    parser.add_argument("--modelpath", type=str, default="bert-base-uncased",
                        help="For distributed training: local_rank")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="accumulation steps for gradient")
    parser.add_argument('--decode_first_K', type=int, default=10000, help="For debugging purpose")
    args = parser.parse_args()

    args.device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    if args.model == 'gpt2-medium':
        args.batch_size = 2
    else:
        args.batch_size = 5

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model = nn.DataParallel(model)
    model.to(args.device)

    if not os.path.exists(args.id):
        os.mkdir(args.id)

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    if args.do_train:
        tb_writer = SummaryWriter(log_dir='tensorboard/GPT2-{}'.format(args.model))
        dataset = GPTTableDatabase('data/train_lm.json', None, None, tokenizer, args.batch_size, args.max_len)        
        model.train()
        optimizer = optim.Adam(model.parameters(), args.learning_rate)

        avg_loss = 0
        global_step = 0
        for epoch_idx in range(args.epoch):
            print("start training {}th epoch".format(epoch_idx))
            dataset.shuffle()
            for idx in range(0, dataset.train_len()):
                batch = dataset.get_data(idx)
                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch
                inputs = torch.cat([caption, trg_inp], 1)

                model.zero_grad()
                optimizer.zero_grad()

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()

                avg_loss += loss.item()

                loss.backward()
                optimizer.step()
                global_step += 1

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("perplexity", math.exp(avg_loss / args.every), global_step)

                    fake_inputs = caption
                    gt_inputs = trg_out.cpu().data.numpy()

                    #samples = model.sample(fake_inputs, tabfeat, caption, highlight_idx, bert)
                    samples = sample_sequence(model, 30, fake_inputs, [])
                    samples = samples[:, caption.shape[1]:]
                    samples = samples.cpu().data.numpy()

                    for s, gt in zip(samples, gt_inputs):
                        text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                        text = text[: text.find(tokenizer.eos_token)]
                        print("PREDICTION |||||| ", text)
                        text = tokenizer.decode(gt, clean_up_tokenization_spaces=True)
                        text = text[: text.find(tokenizer.eos_token)]
                        print("GROUNDTRUH |||||| ",text)
                        break

                    avg_loss = 0

            if args.model == 'gpt2':
                torch.save(model.state_dict(), '{}/GPT_ep{}.pt'.format(args.id, epoch_idx))
            else:
                torch.save(model.state_dict(), '{}/GPT_medium_ep{}.pt'.format(args.id, epoch_idx))
        tb_writer.close()

    if args.do_val:
        dataset = GPTTableDatabase(None, 'data/val_lm.json', None, tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        with torch.no_grad():
            losses = []
            for idx in range(0, dataset.val_len()):
                batch = dataset.get_data(idx, 'val')
                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                inputs = torch.cat([caption, trg_inp], 1)

                logits = model(inputs)[0]
                logits = logits[:, -trg_out.shape[1]:, :].contiguous()

                loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

                loss = loss * mask.view(-1)
                loss = loss.sum() / mask.sum()

                losses.append(loss.item())

                avg_loss = sum(losses) / len(losses)
                perpelexity = math.exp(avg_loss)

                sys.stdout.write("validation perplexity is {} \r".format(perpelexity))

            avg_loss = sum(losses) / len(losses)
            perplexity = math.exp(avg_loss)

            print("validation perplexity is {}".format(perplexity))

    if args.do_test:
        dataset = GPTTableDatabase(None, None, 'data/test_lm.json', tokenizer, args.batch_size, args.max_len)
        model.load_state_dict(torch.load(args.load_from))
        model.eval()

        list_of_hypothesis = []
        list_of_references = []
        results = {}
        with torch.no_grad():
            for idx in range(0, min(args.decode_first_K, dataset.test_len())):
                batch = dataset.get_data(idx, 'test')
                references = dataset.get_reference(idx, 'test')
                table_id = dataset.get_table_id(idx, 'test')
                results[table_id] = []

                batch = tuple(Variable(t).to(device) for t in batch)
                trg_inp, trg_out, mask, caption = batch

                fake_inputs = caption

                samples = sample_sequence(model, 30, fake_inputs, [], top_k=1)

                samples = samples[:, caption.shape[1]:]
                samples = samples.cpu().data.numpy()

                for s in samples:
                    text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    text = text[: text.find(tokenizer.eos_token)]
                    results[table_id].append(text)

                    hypothesis = text.lower().split(' ')
                    
                    list_of_hypothesis.append(hypothesis)
                    list_of_references.append(references)

                bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
                sys.stdout.write('finished {}/{}; BLEU {} \r'.format(idx, dataset.test_len(), bleu))

            bleu = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypothesis)
            print('Overall Corpus BLEU {}'.format(bleu))

        with open('outputs/GPT_{}_{}.json'.format(args.model, bleu), 'w') as f:
            json.dump(results, f, indent=2)
