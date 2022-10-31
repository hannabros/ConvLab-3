# -*- coding: utf-8 -*-
# Copyright 2021 DSML Group, Heinrich Heine University, Düsseldorf
# Authors: Carel van Niekerk (niekerk@hhu.de)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run SetSUMBT Calibration"""

import logging
import random
import os
from shutil import copy2 as copy

import torch
from transformers import (BertModel, BertConfig, BertTokenizer,
                          RobertaModel, RobertaConfig, RobertaTokenizer,
                          AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from convlab.dst.setsumbt.modeling.bert_nbt import BertSetSUMBT
from convlab.dst.setsumbt.modeling.roberta_nbt import RobertaSetSUMBT
from convlab.dst.setsumbt.multiwoz import multiwoz21
from convlab.dst.setsumbt.multiwoz import ontology as embeddings
from convlab.dst.setsumbt.utils import get_args, upload_local_directory_to_gcs, update_args
from convlab.dst.setsumbt.modeling import calibration_utils
from convlab.dst.setsumbt.modeling import ensemble_utils
from convlab.dst.setsumbt.loss.ece import ece, jg_ece, l2_acc


# Datasets
DATASETS = {
    'multiwoz21': multiwoz21
}

MODELS = {
    'bert': (BertSetSUMBT, BertModel, BertConfig, BertTokenizer),
    'roberta': (RobertaSetSUMBT, RobertaModel, RobertaConfig, RobertaTokenizer)
}


def main(args=None, config=None):
    # Get arguments
    if args is None:
        args, config = get_args(MODELS)

    # Select Dataset object
    if args.dataset in DATASETS:
        Dataset = DATASETS[args.dataset]
    else:
        raise NameError('NotImplemented')

    if args.model_type in MODELS:
        SetSumbtModel, CandidateEncoderModel, ConfigClass, Tokenizer = MODELS[args.model_type]
    else:
        raise NameError('NotImplemented')

    # Set up output directory
    OUTPUT_DIR = args.output_dir
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    args.output_dir = OUTPUT_DIR
    if not os.path.exists(os.path.join(OUTPUT_DIR, 'predictions')):
        os.mkdir(os.path.join(OUTPUT_DIR, 'predictions'))

    paths = os.listdir(args.output_dir) if os.path.exists(
        args.output_dir) else []
    if 'pytorch_model.bin' in paths and 'config.json' in paths:
        args.model_name_or_path = args.output_dir
        config = ConfigClass.from_pretrained(args.model_name_or_path)
    else:
        paths = os.listdir(args.output_dir) if os.path.exists(
            args.output_dir) else []
        paths = [os.path.join(args.output_dir, p)
                 for p in paths if 'checkpoint-' in p]
        if paths:
            paths = paths[0]
            args.model_name_or_path = paths
            config = ConfigClass.from_pretrained(args.model_name_or_path)

    if args.ensemble_size > 0:
        paths = os.listdir(args.output_dir) if os.path.exists(
            args.output_dir) else []
        paths = [os.path.join(args.output_dir, p)
                 for p in paths if 'ensemble_' in p]
        if paths:
            args.model_name_or_path = args.output_dir
            config = ConfigClass.from_pretrained(args.model_name_or_path)

    args = update_args(args, config)

    # Set up data directory
    DATA_DIR = args.data_dir
    Dataset.set_datadir(DATA_DIR)
    embeddings.set_datadir(DATA_DIR)

    if args.shrink_active_domains and args.dataset == 'multiwoz21':
        Dataset.set_active_domains(
            ['attraction', 'hotel', 'restaurant', 'taxi', 'train'])

    # Download and preprocess
    Dataset.create_examples(
        args.max_turn_len, args.predict_intents, args.force_processing)

    # Create logger
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if 'stream' not in args.logging_path:
        fh = logging.FileHandler(args.logging_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        ch = logging.StreamHandler()
        ch.setLevel(level=logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Get device
    if torch.cuda.is_available() and args.n_gpu > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        args.n_gpu = 0

    if args.n_gpu == 0:
        args.fp16 = False

    # Set up model training/evaluation
    calibration.set_logger(logger, None)
    calibration.set_seed(args)

    if args.ensemble_size > 0:
        ensemble.set_logger(logger, tb_writer)
        ensemble_utils.set_seed(args)

    # Perform tasks

    if os.path.exists(os.path.join(OUTPUT_DIR, 'predictions', 'test.predictions')):
        pred = torch.load(os.path.join(
            OUTPUT_DIR, 'predictions', 'test.predictions'))
        labels = pred['labels']
        belief_states = pred['belief_states']
        if 'request_labels' in pred:
            request_labels = pred['request_labels']
            request_belief = pred['request_belief']
            domain_labels = pred['domain_labels']
            domain_belief = pred['domain_belief']
            greeting_labels = pred['greeting_labels']
            greeting_belief = pred['greeting_belief']
        else:
            request_belief = None
        del pred
    elif args.ensemble_size > 0:
        # Get training batch loaders and ontology embeddings
        if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'test.db')):
            test_slots = torch.load(os.path.join(
                OUTPUT_DIR, 'database', 'test.db'))
        else:
            # Create Tokenizer and embedding model for Data Loaders and ontology
            encoder = CandidateEncoderModel.from_pretrained(
                config.candidate_embedding_model_name)
            tokenizer = Tokenizer(config.candidate_embedding_model_name)
            embeddings.get_slot_candidate_embeddings(
                'test', args, tokenizer, encoder)
            test_slots = torch.load(os.path.join(
                OUTPUT_DIR, 'database', 'test.db'))

        exists = False
        if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader')):
            test_dataloader = torch.load(os.path.join(
                OUTPUT_DIR, 'dataloaders', 'test.dataloader'))
            if test_dataloader.batch_size == args.test_batch_size:
                exists = True
        if not exists:
            tokenizer = Tokenizer(config.candidate_embedding_model_name)
            test_dataloader = Dataset.get_dataloader('test', args.test_batch_size, tokenizer, args.max_dialogue_len,
                                                     config.max_turn_len)
            torch.save(test_dataloader, os.path.join(
                OUTPUT_DIR, 'dataloaders', 'test.dataloader'))

        config, models = ensemble.get_models(
            args.model_name_or_path, device, ConfigClass, SetSumbtModel)

        belief_states, labels = ensemble_utils.get_predictions(
            args, models, device, test_dataloader, test_slots)
        torch.save({'belief_states': belief_states, 'labels': labels},
                   os.path.join(OUTPUT_DIR, 'predictions', 'test.predictions'))
    else:
        # Get training batch loaders and ontology embeddings
        if os.path.exists(os.path.join(OUTPUT_DIR, 'database', 'test.db')):
            test_slots = torch.load(os.path.join(
                OUTPUT_DIR, 'database', 'test.db'))
        else:
            # Create Tokenizer and embedding model for Data Loaders and ontology
            encoder = CandidateEncoderModel.from_pretrained(
                config.candidate_embedding_model_name)
            tokenizer = Tokenizer(config.candidate_embedding_model_name)
            embeddings.get_slot_candidate_embeddings(
                'test', args, tokenizer, encoder)
            test_slots = torch.load(os.path.join(
                OUTPUT_DIR, 'database', 'test.db'))

        exists = False
        if os.path.exists(os.path.join(OUTPUT_DIR, 'dataloaders', 'test.dataloader')):
            test_dataloader = torch.load(os.path.join(
                OUTPUT_DIR, 'dataloaders', 'test.dataloader'))
            if test_dataloader.batch_size == args.test_batch_size:
                exists = True
        if not exists:
            tokenizer = Tokenizer(config.candidate_embedding_model_name)
            test_dataloader = Dataset.get_dataloader('test', args.test_batch_size, tokenizer, args.max_dialogue_len,
                                                     config.max_turn_len)
            torch.save(test_dataloader, os.path.join(
                OUTPUT_DIR, 'dataloaders', 'test.dataloader'))

        # Initialise Model
        model = SetSumbtModel.from_pretrained(
            args.model_name_or_path, config=config)
        model = model.to(device)

        # Get slot and value embeddings
        slots = {slot: test_slots[slot] for slot in test_slots}
        values = {slot: test_slots[slot][1] for slot in test_slots}

        # Load model ontology
        model.add_slot_candidates(slots)
        for slot in model.informable_slot_ids:
            model.add_value_candidates(slot, values[slot], replace=True)

        belief_states = calibration.get_predictions(
            args, model, device, test_dataloader)
        belief_states, labels, request_belief, request_labels, domain_belief, domain_labels, greeting_belief, greeting_labels = belief_states
        out = {'belief_states': belief_states, 'labels': labels,
               'request_belief': request_belief, 'request_labels': request_labels,
               'domain_belief': domain_belief, 'domain_labels': domain_labels,
               'greeting_belief': greeting_belief, 'greeting_labels': greeting_labels}
        torch.save(out, os.path.join(
            OUTPUT_DIR, 'predictions', 'test.predictions'))

    # err = [ece(belief_states[slot].reshape(-1, belief_states[slot].size(-1)), labels[slot].reshape(-1), 10)
    #         for slot in belief_states]
    # err = max(err)
    # logger.info('ECE: %f' % err)

    # Calculate calibration metrics

    jg = jg_ece(belief_states, labels, 10)
    logger.info('Joint Goal ECE: %f' % jg)

    binary_states = {}
    for slot, p in belief_states.items():
        shp = p.shape
        p = p.reshape(-1, p.size(-1))
        p_ = torch.ones(p.shape).to(p.device) * 1e-8
        p_[range(p.size(0)), p.argmax(-1)] = 1.0 - 1e-8
        binary_states[slot] = p_.reshape(shp)
    jg = jg_ece(binary_states, labels, 10)
    logger.info('Joint Goal Binary ECE: %f' % jg)

    bs = {slot: torch.cat((p[:, :, 0].unsqueeze(-1), p[:, :, 1:].max(-1)
                          [0].unsqueeze(-1)), -1) for slot, p in belief_states.items()}
    ls = {}
    for slot, l in labels.items():
        y = torch.zeros((l.size(0), l.size(1))).to(l.device)
        dials, turns = torch.where(l > 0)
        y[dials, turns] = 1.0
        dials, turns = torch.where(l < 0)
        y[dials, turns] = -1.0
        ls[slot] = y

    jg = jg_ece(bs, ls, 10)
    logger.info('Slot presence ECE: %f' % jg)

    binary_states = {}
    for slot, p in bs.items():
        shp = p.shape
        p = p.reshape(-1, p.size(-1))
        p_ = torch.ones(p.shape).to(p.device) * 1e-8
        p_[range(p.size(0)), p.argmax(-1)] = 1.0 - 1e-8
        binary_states[slot] = p_.reshape(shp)
    jg = jg_ece(binary_states, ls, 10)
    logger.info('Slot presence Binary ECE: %f' % jg)

    jg_acc = 0.0
    padding = torch.cat([item.unsqueeze(-1)
                        for _, item in labels.items()], -1).sum(-1) * -1.0
    padding = (padding == len(labels))
    padding = padding.reshape(-1)
    for slot in belief_states:
        topn = args.accuracy_topn
        p_ = belief_states[slot]
        gold = labels[slot]

        if p_.size(-1) <= topn:
            topn = p_.size(-1) - 1
        if topn <= 0:
            topn = 1

        if topn > 1:
            labs = p_.reshape(-1, p_.size(-1)).argsort(dim=-1, descending=True)
            labs = labs[:, :topn]
        else:
            labs = p_.reshape(-1, p_.size(-1)).argmax(dim=-1).unsqueeze(-1)
        acc = [lab in s for lab, s, pad in zip(
            gold.reshape(-1), labs, padding) if not pad]
        acc = torch.tensor(acc).float()

        jg_acc += acc

    n_turns = jg_acc.size(0)
    sl_acc = sum(jg_acc / len(belief_states)).float()
    jg_acc = sum((jg_acc / len(belief_states)).int()).float()

    sl_acc /= n_turns
    jg_acc /= n_turns

    logger.info('Joint Goal Accuracy: %f, Slot Accuracy %f' % (jg_acc, sl_acc))

    l2 = l2_acc(belief_states, labels, remove_belief=False)
    logger.info(f'Model L2 Norm Goal Accuracy: {l2}')
    l2 = l2_acc(belief_states, labels, remove_belief=True)
    logger.info(f'Binary Model L2 Norm Goal Accuracy: {l2}')

    for slot in belief_states:
        p = belief_states[slot]
        p = p.reshape(-1, p.size(-1))
        p = torch.cat(
            (p[:, 0].unsqueeze(-1), p[:, 1:].max(-1)[0].unsqueeze(-1)), -1)
        belief_states[slot] = p

        l = labels[slot].reshape(-1)
        l[l > 0] = 1
        labels[slot] = l

    f1 = 0.0
    for slot in belief_states:
        prd = belief_states[slot].argmax(-1)
        tp = ((prd == 1) * (labels[slot] == 1)).sum()
        fp = ((prd == 1) * (labels[slot] == 0)).sum()
        fn = ((prd == 0) * (labels[slot] == 1)).sum()
        if tp > 0:
            f1 += tp / (tp + 0.5 * (fp + fn))
    f1 /= len(belief_states)
    logger.info(f'Trucated Goal F1 Score: {f1}')

    l2 = l2_acc(belief_states, labels, remove_belief=False)
    logger.info(f'Model L2 Norm Trucated Goal Accuracy: {l2}')
    l2 = l2_acc(belief_states, labels, remove_belief=True)
    logger.info(f'Binary Model L2 Norm Trucated Goal Accuracy: {l2}')

    if request_belief is not None:
        tp, fp, fn = 0.0, 0.0, 0.0
        for slot in request_belief:
            p = request_belief[slot]
            l = request_labels[slot]

            tp += (p.round().int() * (l == 1)).reshape(-1).float()
            fp += (p.round().int() * (l == 0)).reshape(-1).float()
            fn += ((1 - p.round().int()) * (l == 1)).reshape(-1).float()
        tp /= len(request_belief)
        fp /= len(request_belief)
        fn /= len(request_belief)
        f1 = tp.sum() / (tp.sum() + 0.5 * (fp.sum() + fn.sum()))
        logger.info('Request F1 Score: %f' % f1.item())

        for slot in request_belief:
            p = request_belief[slot]
            p = p.unsqueeze(-1)
            p = torch.cat((1 - p, p), -1)
            request_belief[slot] = p
        jg = jg_ece(request_belief, request_labels, 10)
        logger.info('Request Joint Goal ECE: %f' % jg)

        binary_states = {}
        for slot, p in request_belief.items():
            shp = p.shape
            p = p.reshape(-1, p.size(-1))
            p_ = torch.ones(p.shape).to(p.device) * 1e-8
            p_[range(p.size(0)), p.argmax(-1)] = 1.0 - 1e-8
            binary_states[slot] = p_.reshape(shp)
        jg = jg_ece(binary_states, request_labels, 10)
        logger.info('Request Joint Goal Binary ECE: %f' % jg)

        tp, fp, fn = 0.0, 0.0, 0.0
        for dom in domain_belief:
            p = domain_belief[dom]
            l = domain_labels[dom]

            tp += (p.round().int() * (l == 1)).reshape(-1).float()
            fp += (p.round().int() * (l == 0)).reshape(-1).float()
            fn += ((1 - p.round().int()) * (l == 1)).reshape(-1).float()
        tp /= len(domain_belief)
        fp /= len(domain_belief)
        fn /= len(domain_belief)
        f1 = tp.sum() / (tp.sum() + 0.5 * (fp.sum() + fn.sum()))
        logger.info('Domain F1 Score: %f' % f1.item())

        for dom in domain_belief:
            p = domain_belief[dom]
            p = p.unsqueeze(-1)
            p = torch.cat((1 - p, p), -1)
            domain_belief[dom] = p
        jg = jg_ece(domain_belief, domain_labels, 10)
        logger.info('Domain Joint Goal ECE: %f' % jg)

        binary_states = {}
        for slot, p in domain_belief.items():
            shp = p.shape
            p = p.reshape(-1, p.size(-1))
            p_ = torch.ones(p.shape).to(p.device) * 1e-8
            p_[range(p.size(0)), p.argmax(-1)] = 1.0 - 1e-8
            binary_states[slot] = p_.reshape(shp)
        jg = jg_ece(binary_states, domain_labels, 10)
        logger.info('Domain Joint Goal Binary ECE: %f' % jg)

        tp = ((greeting_belief.argmax(-1) > 0) *
              (greeting_labels > 0)).reshape(-1).float().sum()
        fp = ((greeting_belief.argmax(-1) > 0) *
              (greeting_labels == 0)).reshape(-1).float().sum()
        fn = ((greeting_belief.argmax(-1) == 0) *
              (greeting_labels > 0)).reshape(-1).float().sum()
        f1 = tp / (tp + 0.5 * (fp + fn))
        logger.info('Greeting F1 Score: %f' % f1.item())

        err = ece(greeting_belief.reshape(-1, greeting_belief.size(-1)),
                  greeting_labels.reshape(-1), 10)
        logger.info('Greetings ECE: %f' % err)

        greeting_belief = greeting_belief.reshape(-1, greeting_belief.size(-1))
        binary_states = torch.ones(greeting_belief.shape).to(
            greeting_belief.device) * 1e-8
        binary_states[range(greeting_belief.size(0)),
                      greeting_belief.argmax(-1)] = 1.0 - 1e-8
        err = ece(binary_states, greeting_labels.reshape(-1), 10)
        logger.info('Greetings Binary ECE: %f' % err)

        for slot in request_belief:
            p = request_belief[slot].unsqueeze(-1)
            request_belief[slot] = torch.cat((1 - p, p), -1)

        l2 = l2_acc(request_belief, request_labels, remove_belief=False)
        logger.info(f'Model L2 Norm Request Accuracy: {l2}')
        l2 = l2_acc(request_belief, request_labels, remove_belief=True)
        logger.info(f'Binary Model L2 Norm Request Accuracy: {l2}')

        for slot in domain_belief:
            p = domain_belief[slot].unsqueeze(-1)
            domain_belief[slot] = torch.cat((1 - p, p), -1)

        l2 = l2_acc(domain_belief, domain_labels, remove_belief=False)
        logger.info(f'Model L2 Norm Domain Accuracy: {l2}')
        l2 = l2_acc(domain_belief, domain_labels, remove_belief=True)
        logger.info(f'Binary Model L2 Norm Domain Accuracy: {l2}')

        greeting_labels = {'bye': greeting_labels}
        greeting_belief = {'bye': greeting_belief}

        l2 = l2_acc(greeting_belief, greeting_labels, remove_belief=False)
        logger.info(f'Model L2 Norm Greeting Accuracy: {l2}')
        l2 = l2_acc(greeting_belief, greeting_labels, remove_belief=False)
        logger.info(f'Binary Model L2 Norm Greeting Accuracy: {l2}')


if __name__ == "__main__":
    main()
