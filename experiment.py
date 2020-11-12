from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

from config import config_args
from train import train
import argparse
from utils.train_utils import add_flags_from_config
from copy import deepcopy


def get_args(model, manifold, dim, dataset, log_freq,
             cuda, lr, n_layers, act, bias, dropout,
             weight_decay, c, normalize_feats, task,
             mixed_fractions, lr_reduce_freq):
    cfg = deepcopy(config_args)

    cfg['model_config']['model'] = (model,"")
    cfg['model_config']['manifold'] = (manifold,"")
    cfg['model_config']['dim'] = (dim,"")
    cfg['model_config']['num-layers'] = (n_layers,"")
    cfg['model_config']['act'] = (act,"")
    cfg['model_config']['bias'] = (bias,"")
    if manifold == "Hyperboloid":
        cfg['model_config']['mixed_frac'] = ([1,0,0], "")
    if manifold == "Euclidean":
        cfg['model_config']['mixed_frac'] = ([0,1,0], "")
    if manifold == "PoincareBall":
        cfg['model_config']['mixed_frac'] = ([0,0,1], "")
    else:
        cfg['model_config']['mixed_frac'] = (mixed_fractions, "")
    if c is not None:
        cfg['model_config']['c'] = (float(c),"")
    else:
        cfg['model_config']['c'] = (c,"")
    cfg['model_config']['task'] = (task,"")

    cfg['training_config']['cuda'] = (cuda,"")
    cfg['training_config']['log-freq'] = (log_freq,"")
    cfg['training_config']['lr'] = (lr,"")
    cfg['training_config']['lr-reduce-freq'] = (lr_reduce_freq,"")
    cfg['training_config']['dropout'] = (float(dropout),"")
    cfg['training_config']['weight-decay'] = (float(weight_decay),"")

    cfg['data_config']['dataset'] = (dataset,"")
    cfg['data_config']['normalize-feats'] = (float(normalize_feats),"")

    # unnecessary now
    # if manifold == "Mixture" and mixed_fractions[0]==1:
    #     cfg['model_config']['manifold'] = ("Hyperboloid","")
    # if manifold == "Mixture" and mixed_fractions[1]==1:
    #     cfg['model_config']['manifold'] = ("Euclidean","")
    # if manifold == "Mixture" and mixed_fractions[2]==1:
    #     cfg['model_config']['manifold'] = ("PoincareBall","")

    parser = argparse.ArgumentParser()
    for _, config_dict in cfg.items():
        parser = add_flags_from_config(parser, config_dict)
    args = parser.parse_args([])

    return args


def run_experiment(model, manifold, dim, dataset="cora", log_freq=5, cuda=-1,
                   lr=0.01, n_layers=2, act="relu", bias=1, dropout=0.5,
                   weight_decay=0.001, c=None, normalize_feats=1, task="lp",
                   mixed_fractions=[1/3,1/3,1/3], lr_reduce_freq=None):
    args = get_args(model, manifold, dim, dataset, log_freq,
                   cuda, lr, n_layers, act, bias, dropout,
                   weight_decay, c, normalize_feats, task,
                   mixed_fractions, lr_reduce_freq)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])
    logging.info(f"{args.model}\t{args.manifold}\t{args.task}\t{args.dataset}")
    logging.info(f"Dim:{args.dim}  lr:{args.lr}  decay:{args.lr_reduce_freq}")
    if args.manifold == "Mixture":
        m = args.mixed_frac
        logging.info(f"Hyperboloid:{100*m[0]:.1f}%  Euclidean:{100*m[1]:.1f}%  PoincareBall:{100*m[2]:.1f}%")
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel
        else:
            Model = RECModel
            # No validation for reconstruction task
            args.eval_freq = args.epochs + 1

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    # logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    history = {"eval_freq":args.eval_freq,
               "model":args.model,
               "dataset":args.dataset,
               "dim":args.dim,
               "manifold":args.manifold,
               "mix_frac":args.mixed_frac,
               "task":args.task}
    for s in ["train", "val", "test"]:
        if args.task == "nc":
            for m in ["acc", "f1", "loss"]:
                history[s+"_"+m] = []
        elif args.task == "lp":
            for m in ["roc", "ap", "loss"]:
                history[s+"_"+m] = []
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        history["train_loss"] += [train_metrics["loss"].item()]
        if args.task == "lp":
            history["train_roc"] += [train_metrics["roc"]]
            history["train_ap"] += [train_metrics["ap"]]
        if args.task == "nc":
            history["train_acc"] += [train_metrics["acc"]]
            history["train_f1"] += [train_metrics["f1"]]
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            embeddings = model.encode(data['features'], data['adj_train_norm'])
            val_metrics = model.compute_metrics(embeddings, data, 'val')
            history["val_loss"] += [val_metrics["loss"].item()]
            if args.task == "lp":
                history["val_roc"] += [val_metrics["roc"]]
                history["val_ap"] += [val_metrics["ap"]]
            if args.task == "nc":
                history["val_acc"] += [val_metrics["acc"]]
                history["val_f1"] += [val_metrics["f1"]]
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            if model.has_improved(best_val_metrics, val_metrics):
                best_test_metrics = model.compute_metrics(embeddings, data, 'test')
                best_emb = embeddings.cpu()
                if args.save:
                    np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.detach().numpy())
                best_val_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info(f"Early stopping after {epoch} epochs.")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
    history["test_loss"] = best_test_metrics["loss"].item()
    if args.task == "lp":
        history["test_roc"] = best_test_metrics["roc"]
        history["test_ap"] = best_test_metrics["ap"]
    if args.task == "nc":
        history["test_acc"] = best_test_metrics["acc"]
        history["test_f1"] = best_test_metrics["f1"]
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")
    logging.shutdown()
    print("\n")
    return history

