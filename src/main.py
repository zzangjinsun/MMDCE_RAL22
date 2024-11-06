"""
    Adaptive Cost Volume Fusion Network for Multi-Modal Depth Estimation in Changing Environments
    Jinsun Park, Yongseop Jeong, Kyungdon Joo, Donghyeon Cho and In So Kweon
    IEEE Robotics and Automation Letters (RAL), Feb 2022
    IEEE International Conference on Robotics and Automation (ICRA), May 2022
    Project Page: https://github.com/zzangjinsun/MMDCE_RAL22
    Author: Jinsun Park (jspark@pusan.ac.kr)

    ======================================================================

    The main script for training and testing.
"""

from config import args as args_config
import time
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = args_config.port

import json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utility
from model import get as get_model
from data import get as get_data
from loss import get as get_loss
from summary import get as get_summary
from metric import get as get_metric

# Multi-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist

# Minimize randomness
torch.manual_seed(args_config.seed)
np.random.seed(args_config.seed)
random.seed(args_config.seed)
torch.cuda.manual_seed_all(args_config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def check_args(args):
    if args.batch_size < args.num_gpus:
        print("batch_size changed : {} -> {}".format(args.batch_size,
                                                     args.num_gpus))
        args.batch_size = args.num_gpus

    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain)

            new_args = checkpoint['args']
            new_args.test_only = args.test_only
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume

    return new_args


def train(gpu, args):
    # Initialize workers
    # NOTE : the worker with gpu=0 will do logging
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    # Prepare dataset
    data = get_data(args)

    data_train = data(args, 'train')
    data_val = data(args, 'val')

    sampler_train = DistributedSampler(
        data_train, num_replicas=args.num_gpus, rank=gpu)
    sampler_val = DistributedSampler(
        data_val, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size // args.num_gpus

    loader_train = DataLoader(
        dataset=data_train, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    loader_val = DataLoader(
        dataset=data_val, batch_size=1, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_val,
        drop_last=False)

    # Network
    model = get_model(args)
    net = model(args)
    net.cuda(gpu)

    if gpu == 0:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain)
            net.load_state_dict(checkpoint['net'])

            print('Load network parameters from : {}'.format(args.pretrain))

    # Loss
    loss = get_loss(args)
    loss = loss(args)
    loss.cuda(gpu)

    # Optimizer
    optimizer, scheduler = utility.make_optimizer_scheduler(args, net)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])

                    print('Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. '
                          'Use --save_full argument')

            del checkpoint

    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu], output_device=gpu)

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    if gpu == 0:
        utility.backup_source_code(args.save_dir + '/code')
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            os.makedirs(args.save_dir + '/train', exist_ok=True)
            os.makedirs(args.save_dir + '/val', exist_ok=True)
        except OSError:
            pass

    if gpu == 0:
        writer_train = summary(args.save_dir, 'train', args,
                               loss.loss_name, metric.metric_name)
        writer_val = summary(args.save_dir, 'val', args,
                             loss.loss_name, metric.metric_name)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)

    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    if args.save_best != 'none':
        best_epoch = -1

        if args.save_best == 'max':
            best_metric = -1e10
        elif args.save_best == 'min':
            best_metric = 1e10
        else:
            raise NotImplementedError

    for epoch in range(1, args.epochs+1):
        # Train
        net.train()

        sampler_train.set_epoch(epoch)

        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])

            print('=== Epoch {:5d} / {:5d} | Lr : {} | {} | {} ==='.format(
                epoch, args.epochs, list_lr, current_time, args.save_dir
            ))

        num_sample = len(data_train)

        if gpu == 0:
            pbar = tqdm(total=num_sample, dynamic_ncols=True)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if val is not None}

            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] \
                                 * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            loss_sum.backward()

            optimizer.step()

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_train.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Train', current_time, log_loss / log_cnt)

                if epoch == 1 and args.warm_up:
                    list_lr = []
                    for g in optimizer.param_groups:
                        list_lr.append(round(g['lr'], 6))
                    error_str = '{} | Lr Warm Up : {}'.format(error_str,
                                                              list_lr)

                pbar.set_description(error_str)
                pbar.update(loader_train.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            writer_train.update(epoch, sample, output)

            if args.save_full or epoch == args.epochs:
                state = {
                    'net': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'args': args
                }
            else:
                state = {
                    'net': net.module.state_dict(),
                    'args': args
                }

            torch.save(state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))

        # Val
        torch.set_grad_enabled(False)
        net.eval()

        num_sample = len(data_val)

        if gpu == 0:
            pbar = tqdm(total=num_sample, dynamic_ncols=True)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(loader_val):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                      if val is not None}

            output = net(sample)

            loss_sum, loss_val = loss(sample, output)

            # Divide by batch size
            loss_sum = loss_sum / loader_val.batch_size
            loss_val = loss_val / loader_val.batch_size

            if gpu == 0:
                metric_val = metric.evaluate(sample, output, 'train')
                writer_val.add(loss_val, metric_val)

                log_cnt += 1
                log_loss += loss_sum.item()

                current_time = time.strftime('%y%m%d@%H:%M:%S')
                error_str = '{:<10s}| {} | Loss = {:.4f}'.format(
                    'Val', current_time, log_loss / log_cnt)
                pbar.set_description(error_str)
                pbar.update(loader_val.batch_size * args.num_gpus)

        if gpu == 0:
            pbar.close()

            if args.save_best != 'none':
                flag_best_updated = False
                
                metric_val_all = np.concatenate(writer_val.metric, axis=0)
                metric_val_all = np.mean(metric_val_all, axis=0, keepdims=False)

                # NOTE: The best ckpt is determined based on the first metric
                if args.save_best == 'max':
                    if metric_val_all[0] > best_metric:
                        best_epoch = epoch
                        best_metric = metric_val_all[0]
                        flag_best_updated = True
                elif args.save_best == 'min':
                    if metric_val_all[0] < best_metric:
                        best_epoch = epoch
                        best_metric = metric_val_all[0]
                        flag_best_updated = True
                else:
                    raise NotImplementedError

            writer_val.update(epoch, sample, output)

            writer_val.save(epoch, batch, sample, output)
            
            if args.save_best != 'none':
                if flag_best_updated:
                    torch.save(state, '{}/best.pt'.format(args.save_dir))
                    print('          | Best checkpoint updated ({}: {:.5f})'.format(metric.metric_name[0], best_metric))
                
            print('')

        torch.set_grad_enabled(True)

        scheduler.step()


def test(args):
    # Prepare dataset
    data = get_data(args)

    data_test = data(args, 'test')

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # Network
    model = get_model(args)
    net = model(args)
    net.cuda()

    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)
        
        print('Test with the checkpoint: {}'.format(args.pretrain))

        checkpoint = torch.load(args.pretrain)
        key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

        if key_u:
            print('Unexpected keys :')
            print(key_u)

        if key_m:
            print('Missing keys :')
            print(key_m)
            raise KeyError

    net = nn.DataParallel(net)

    metric = get_metric(args)
    metric = metric(args)
    summary = get_summary(args)

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/test', exist_ok=True)
    except OSError:
        pass

    writer_test = summary(args.save_dir, 'test', args, None, metric.metric_name)

    torch.set_grad_enabled(False)
    net.eval()

    num_sample = len(data_test)

    pbar = tqdm(total=num_sample, dynamic_ncols=True)

    t_total = 0

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if val is not None}

        t0 = time.time()
        output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'train')

        writer_test.add(None, metric_val)

        # Save data for analysis
        if args.save_image:
            writer_test.save(args.epochs, batch, sample, output)

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{:<10s}| {} '.format('Test', current_time)
        pbar.set_description(error_str)
        pbar.update(loader_test.batch_size)

    pbar.close()

    writer_test.update(args.epochs, sample, output)

    t_avg = t_total / num_sample
    print('Elapsed time : {:.4f} sec, '
          'Average processing time : {:.4f} sec'.format(t_total, t_avg))


def main(args):
    if not args.test_only:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        if args.save_best == 'none':
            args.pretrain = '{}/model_{:05d}.pt'.format(args.save_dir,
                                                        args.epochs)
        else:
            args.pretrain = '{}/best.pt'.format(args.save_dir)

    test(args)


if __name__ == '__main__':
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        if (cnt + 1) % 5 == 0:
            print('')
        cnt += 1
    print('\n')

    main(args_main)
