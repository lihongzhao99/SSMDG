import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from mmaction.apis import init_recognizer
from torch.cuda.amp import GradScaler
from VGGSound.model import AVENet
from VGGSound.models.resnet import AudioAttGenModule
from VGGSound.test import get_arguments

from dataloader_EPIC_semi import UnifiedEPICDomainSemiDataset, UnifiedSemiDataLoader, create_unified_datasets
from semi_train_utils import (
    DEFAULT_NUM_CLASSES,
    Encoder,
    EncoderTrans,
    calculate_per_class_metrics,
    save_best_model,
    setup_prototype_manager,
    train_one_step_base,
    validate_one_step,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    domain_group = parser.add_argument_group('Domain and data')
    domain_group.add_argument('-s', '--source_domain', nargs='+', required=True, help='Source domains')
    domain_group.add_argument('-t', '--target_domain', nargs='+', required=True, help='Target domains')
    domain_group.add_argument('--datapath', type=str, required=True, help='Dataset path')

    semi_group = parser.add_argument_group('Semi-supervised setting')
    semi_group.add_argument('--semi_setting', type=str,
                            choices=['number', 'ratio', 'domain'],
                            help='Semi-supervised setting: number, ratio, or domain')
    semi_group.add_argument('--semi_value', type=float,
                            help='Semi-supervised value: int for number, float (0-1) for ratio, unused for domain')
    semi_group.add_argument('--unlabeled_domains', nargs='+',
                            help='Domains that are completely unlabeled (only for domain setting)')

    train_group = parser.add_argument_group('Training hyperparameters')
    train_group.add_argument('--optimizer', type=str, default='adamw',
                             choices=['adamw', 'adam', 'sgd'], help='Optimizer type')
    train_group.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_group.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    train_group.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    train_group.add_argument('--bsz', type=int, default=32, help='Batch size')
    train_group.add_argument('--num_workers', type=int, default=8, help='DataLoader worker count')
    train_group.add_argument('--threshold', type=float, default=0.95, help='Pseudo-label confidence threshold')
    train_group.add_argument('--nepochs', type=int, default=50, help='Number of epochs')

    model_group = parser.add_argument_group('Model options')
    model_group.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES, help='Number of action classes')
    model_group.add_argument('--use_video', action='store_true')
    model_group.add_argument('--use_flow', action='store_true')
    model_group.add_argument('--use_audio', action='store_true')

    loss_group = parser.add_argument_group('Loss weights')
    loss_group.add_argument('--lambda_gce', type=float, default=0.1)
    loss_group.add_argument('--lambda_u', type=float, default=1.0)
    loss_group.add_argument('--lambda_mse_proto', type=float, default=0.01)
    loss_group.add_argument('--lambda_cross_modal', type=float, default=0.01)

    runtime_group = parser.add_argument_group('Runtime and logging')
    runtime_group.add_argument('--BestEpoch', type=int, default=0)
    runtime_group.add_argument('--BestAcc', type=float, default=0, help='Best accuracy')
    runtime_group.add_argument('--BestLoss', type=float, default=0, help='Best loss')
    runtime_group.add_argument('--BestTestAcc', type=float, default=0, help='Best test accuracy')
    runtime_group.add_argument('--appen', type=str, default='', help='Append to log name')
    runtime_group.add_argument('--seed', type=int, default=0, help='Random seed')
    runtime_group.add_argument('--max_split_size_mb', type=int, default=128,
                               help='Set PYTORCH_CUDA_ALLOC_CONF max_split_size_mb to reduce CUDA memory fragmentation')

    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{args.max_split_size_mb}"

    if not any([args.use_video, args.use_flow, args.use_audio]):
        parser.error("At least one modality must be enabled: --use_video, --use_flow, or --use_audio")

    if args.semi_setting == 'number' and not isinstance(int(args.semi_value), int):
        parser.error("For 'number' setting, semi_value must be an integer")
    elif args.semi_setting == 'ratio' and not (0.0 <= args.semi_value <= 1.0):
        parser.error("For 'ratio' setting, semi_value must be between 0.0 and 1.0")
    elif args.semi_setting == 'domain' and not args.unlabeled_domains:
        parser.error("For 'domain' setting, unlabeled_domains must be specified")

    if args.semi_setting == 'number':
        args.semi_value = int(args.semi_value)

    args.use_amp = torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda:0'
    device = torch.device(device)

    config_file = os.path.join(
        script_dir,
        'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    )
    checkpoint_file = os.path.join(
        script_dir,
        'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'
    )
    config_file_flow = os.path.join(
        script_dir,
        'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    )
    checkpoint_file_flow = os.path.join(
        script_dir,
        'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'
    )

    input_dim = 0
    cfg = None
    cfg_flow = None

    if args.use_video:
        model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
        model.cls_head.fc_cls = nn.Linear(2304, args.num_classes).cuda()
        cfg = model.cfg
        model = torch.nn.DataParallel(model)
        input_dim += 2304

    if args.use_flow:
        model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device, use_frames=True)
        model_flow.cls_head.fc_cls = nn.Linear(2048, args.num_classes).cuda()
        cfg_flow = model_flow.cfg
        model_flow = torch.nn.DataParallel(model_flow)
        input_dim += 2048

    if args.use_audio:
        audio_args = get_arguments()
        audio_model = AVENet(audio_args)
        checkpoint = torch.load(os.path.join(script_dir, 'pretrained_models/vggsound_avgpool.pth.tar'))
        audio_model.load_state_dict(checkpoint['model_state_dict'])
        audio_model = audio_model.cuda()
        audio_model.eval()

        audio_cls_model = AudioAttGenModule()
        audio_cls_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        audio_cls_model.fc = nn.Linear(512, args.num_classes)
        audio_cls_model = audio_cls_model.cuda()
        input_dim += 512

    mlp_cls = Encoder(input_dim=input_dim, out_dim=args.num_classes).cuda()
    mlp_v2a = EncoderTrans(input_dim=2304, hidden=1028, out_dim=512).cuda() if (args.use_video and args.use_audio) else None
    mlp_a2v = EncoderTrans(input_dim=512, hidden=1028, out_dim=2304).cuda() if (args.use_video and args.use_audio) else None
    mlp_v2f = EncoderTrans(input_dim=2304, hidden=1028, out_dim=2048).cuda() if (args.use_video and args.use_flow) else None
    mlp_f2v = EncoderTrans(input_dim=2048, hidden=1028, out_dim=2304).cuda() if (args.use_video and args.use_flow) else None
    mlp_f2a = EncoderTrans(input_dim=2048, hidden=1028, out_dim=512).cuda() if (args.use_flow and args.use_audio) else None
    mlp_a2f = EncoderTrans(input_dim=512, hidden=1028, out_dim=2048).cuda() if (args.use_flow and args.use_audio) else None

    print("Using regular Encoder classifier")

    models = {
        'model': model if args.use_video else None,
        'model_flow': model_flow if args.use_flow else None,
        'audio_model': audio_model if args.use_audio else None,
        'audio_cls_model': audio_cls_model if args.use_audio else None,
        'mlp_cls': mlp_cls,
        'mlp_v2a': mlp_v2a,
        'mlp_a2v': mlp_a2v,
        'mlp_v2f': mlp_v2f,
        'mlp_f2v': mlp_f2v,
        'mlp_f2a': mlp_f2a,
        'mlp_a2f': mlp_a2f,
    }

    criterion = nn.CrossEntropyLoss().cuda()
    batch_size = args.bsz

    params = list(mlp_cls.parameters())
    if mlp_v2a is not None:
        params += list(mlp_v2a.parameters())
    if mlp_a2v is not None:
        params += list(mlp_a2v.parameters())
    if mlp_v2f is not None:
        params += list(mlp_v2f.parameters())
    if mlp_f2v is not None:
        params += list(mlp_f2v.parameters())
    if mlp_f2a is not None:
        params += list(mlp_f2a.parameters())
    if mlp_a2f is not None:
        params += list(mlp_a2f.parameters())
    if args.use_video:
        params += (list(model.module.backbone.fast_path.layer4.parameters()) +
                   list(model.module.backbone.slow_path.layer4.parameters()) +
                   list(model.module.cls_head.parameters()))
    if args.use_flow:
        params += (list(model_flow.module.backbone.layer4.parameters()) +
                   list(model_flow.module.cls_head.parameters()))
    if args.use_audio:
        params += list(audio_cls_model.parameters())

    if args.optimizer == 'adamw':
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == 'sgd':
        optim = torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    scaler = GradScaler(enabled=args.use_amp)

    BestLoss = float("inf")
    BestEpoch = args.BestEpoch
    BestAcc = args.BestAcc
    BestTestAcc = args.BestTestAcc

    print(f"Training base From Scratch ...")
    starting_epoch = 0

    prototype_manager = setup_prototype_manager(args, num_domains=2)

    labeled_train_dataset, unlabeled_train_dataset = create_unified_datasets(
        cfg, args.source_domain, args.datapath, args.use_video, args.use_audio,
        args.semi_setting, args.semi_value, args.unlabeled_domains,
        cfg_flow=cfg_flow, use_flow=args.use_flow
    )

    train_dataloader = UnifiedSemiDataLoader(
        labeled_train_dataset, unlabeled_train_dataset,
        batch_size=batch_size, num_workers=args.num_workers
    )

    val_dataset = UnifiedEPICDomainSemiDataset(
        split='test', domain=args.source_domain, cfg=cfg, cfg_flow=cfg_flow,
        datapath=args.datapath, use_video=args.use_video,
        use_flow=args.use_flow,
        use_audio=args.use_audio, is_labeled=True,
        semi_setting=args.semi_setting, semi_value=args.semi_value,
        unlabeled_domains=args.unlabeled_domains
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )

    test_dataset = UnifiedEPICDomainSemiDataset(
        split='test', domain=args.target_domain, cfg=cfg, cfg_flow=cfg_flow,
        datapath=args.datapath, use_video=args.use_video,
        use_flow=args.use_flow,
        use_audio=args.use_audio, is_labeled=True,
        semi_setting=args.semi_setting, semi_value=args.semi_value,
        unlabeled_domains=args.unlabeled_domains
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )

    print(f"Labeled samples: {len(labeled_train_dataset)}")
    print(f"Unlabeled samples: {len(unlabeled_train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Training loop
    for epoch_i in range(starting_epoch, args.nepochs):
        print(f"Epoch: {epoch_i:02d}")

        # Training phase
        acc = 0
        count = 0
        total_loss = 0
        total_labeled_loss = 0
        total_pseudo_correct = 0
        total_pseudo_count = 0
        num_batches = 0
        total_initial_pseudo = 0

        total_ce_count = 0
        total_gce_count = 0
        total_unlabeled_samples = 0

        epoch_pseudo_class_stats = {i: {'count': 0, 'correct': 0} for i in range(args.num_classes)}
        epoch_domain_stats = {}

        print('train')
        mlp_cls.train()
        if mlp_v2a is not None:
            mlp_v2a.train()
        if mlp_a2v is not None:
            mlp_a2v.train()
        if mlp_v2f is not None:
            mlp_v2f.train()
        if mlp_f2v is not None:
            mlp_f2v.train()
        if mlp_f2a is not None:
            mlp_f2a.train()
        if mlp_a2f is not None:
            mlp_a2f.train()
        if args.use_video:
            model.train()
        if args.use_flow:
            model_flow.train()
        if args.use_audio:
            audio_cls_model.train()

        with tqdm.tqdm(total=len(train_dataloader)) as pbar:
            for i, (labeled_batch, unlabeled_batch) in enumerate(train_dataloader):
                result = train_one_step_base(
                    labeled_batch, unlabeled_batch, epoch_i,
                    models, criterion, optim, args, prototype_manager, args.threshold, scaler=scaler
                )
                (predict1, loss, labeled_loss, unlabeled_loss,
                 pseudo_acc, pseudo_correct, pseudo_total, per_class_stats,
                 domain_stats, initial_pseudo_count, ce_count, gce_count,
                 batch_unlabeled_size) = result

                batch_size_actual = predict1.size(0)
                total_loss += loss.item() * batch_size_actual
                total_labeled_loss += labeled_loss.item() * batch_size_actual
                total_pseudo_correct += pseudo_correct
                total_pseudo_count += pseudo_total
                num_batches += 1

                total_initial_pseudo += initial_pseudo_count

                total_ce_count += ce_count
                total_gce_count += gce_count
                total_unlabeled_samples += batch_unlabeled_size

                for class_id in range(args.num_classes):
                    epoch_pseudo_class_stats[class_id]['count'] += per_class_stats[class_id]['count']
                    epoch_pseudo_class_stats[class_id]['correct'] += per_class_stats[class_id]['correct']

                for domain_id, stats in domain_stats.items():
                    if domain_id not in epoch_domain_stats:
                        epoch_domain_stats[domain_id] = {'count': 0}
                    epoch_domain_stats[domain_id]['count'] += stats['count']

                _, predict_gpu = torch.max(predict1.detach(), dim=1)
                predict = predict_gpu.cpu()
                labels = labeled_batch[3] if args.use_flow else labeled_batch[2]

                acc1 = (predict == labels).sum().item()
                acc += int(acc1)
                count += batch_size_actual

                avg_pseudo_acc = (
                            total_pseudo_correct / total_pseudo_count) if total_pseudo_count > 0 else 0.0

                avg_ce_ratio = total_ce_count / total_unlabeled_samples if total_unlabeled_samples > 0 else 0.0
                avg_gce_ratio = total_gce_count / total_unlabeled_samples if total_unlabeled_samples > 0 else 0.0

                pbar.set_postfix_str(
                    f"Loss: {total_loss / float(count):.4f}, "
                    f"Acc: {acc / float(count):.4f}, "
                    f"Pseudo: {avg_pseudo_acc:.4f} ({total_pseudo_correct}/{total_pseudo_count}), "
                    f"CE: {avg_ce_ratio:.3f}, "
                    f"GCE: {avg_gce_ratio:.3f}"
                )
                pbar.update()

        train_acc = acc / float(count)
        train_loss = total_loss / float(count)
        train_labeled_loss = total_labeled_loss / float(count)
        train_pseudo_acc = (
                    total_pseudo_correct / total_pseudo_count) if total_pseudo_count > 0 else 0.0
        ce_ratio_epoch = total_ce_count / total_unlabeled_samples if total_unlabeled_samples > 0 else 0.0
        gce_ratio_epoch = total_gce_count / total_unlabeled_samples if total_unlabeled_samples > 0 else 0.0


        print(f"\nEpoch {epoch_i} Training Summary:")
        print(f"  Loss: {train_loss:.4f}")
        print(f"  Accuracy: {train_acc:.4f}")
        print(f"  Pseudo Label Accuracy: {train_pseudo_acc:.4f}")
        print(f"  Relaxed Condition:")
        print(f"    Initial Pseudo-labels: {total_initial_pseudo}")
        print(f"  CE Ratio: {ce_ratio_epoch:.4f}")
        print(f"  GCE Ratio: {gce_ratio_epoch:.4f}")

        if total_pseudo_count > 0:
            print(f"\n[Epoch {epoch_i}] Pseudo-Label Domain Distribution Summary:")
            print("Domain ID | Count | Percentage")
            print("-" * 40)
            for domain_id in sorted(epoch_domain_stats.keys()):
                stats = epoch_domain_stats[domain_id]
                percentage = stats['count'] / total_pseudo_count * 100
                print(f"Domain {domain_id:2d} | {stats['count']:5d} | {percentage:6.2f}%")

        print(f"\nPseudo-Label Per-Class Statistics:")
        print("Class ID | Pseudo Count | Accuracy | Correct/Total")
        print("-" * 55)
        for class_id in range(args.num_classes):
            stats = epoch_pseudo_class_stats[class_id]
            if stats['count'] > 0:
                class_acc = stats['correct'] / stats['count']
            else:
                class_acc = 0.0
            print(f"Class {class_id:2d} | {stats['count']:12d} | {class_acc:8.4f} | "
                  f"{stats['correct']:7d}/{stats['count']:5d}")

        is_best_val = False
        for split, dataloader in [('val', val_dataloader), ('test', test_dataloader)]:
            acc = 0
            count = 0
            total_loss = 0

            all_predictions = []
            all_labels = []

            print(split)

            mlp_cls.eval()
            if args.use_video:
                model.eval()
            if args.use_flow:
                model_flow.eval()
            if args.use_audio:
                audio_cls_model.eval()

            with tqdm.tqdm(total=len(dataloader)) as pbar:
                for (i, batch) in enumerate(dataloader):
                    if args.use_flow:
                        clip, flow, spectrogram, labels, domain_labels = batch
                    else:
                        clip, spectrogram, labels, domain_labels = batch
                        flow = None
                    predict1, loss = validate_one_step(clip, labels, flow, spectrogram, models, criterion, args)

                    batch_size_actual = predict1.size(0)
                    total_loss += loss.item() * batch_size_actual
                    _, predict_gpu = torch.max(predict1.detach(), dim=1)
                    predict = predict_gpu.cpu()

                    all_predictions.append(predict)
                    all_labels.append(labels)

                    acc1 = (predict == labels).sum().item()
                    acc += int(acc1)
                    count += predict1.size()[0]

                    pbar.set_postfix_str(
                        f"Average loss: {total_loss / float(count):.4f}, "
                        f"Current loss: {loss.item():.4f}, "
                        f"Accuracy: {acc / float(count):.4f}"
                    )
                    pbar.update()

            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            class_stats = calculate_per_class_metrics(all_predictions, all_labels, num_classes=args.num_classes)

            print(f"\n{split.upper()} - Per-class statistics for Epoch {epoch_i}:")
            print("Class ID | Sample Count | Accuracy | Correct/Total")
            print("-" * 50)
            for class_id in range(args.num_classes):
                stats = class_stats[class_id]
                print(
                    f"Class {class_id:2d} | {stats['count']:12d} | {stats['accuracy']:8.4f} | {stats['correct']:7d}/{stats['count']:5d}")

            if split == 'val':
                currentvalAcc = acc / float(count)
                currentloss = total_loss / float(count)
                is_best_val = currentvalAcc >= BestAcc
                if is_best_val:
                    BestEpoch = epoch_i
                    BestAcc = acc / float(count)

                    save_best_model(
                        epoch_i,
                        models,
                        optim,
                        BestAcc,
                        BestLoss
                    )

                if currentloss <= BestLoss:
                    BestLoss = total_loss / float(count)

            if split == 'test':
                currenttestAcc = acc / float(count)
                if is_best_val:
                    BestTestAcc = currenttestAcc

        print(f'\nEpoch {epoch_i} Results:')
        print(
            f'Base Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, L_Loss: {train_labeled_loss:.4f}')
        print(f'Val - Acc: {currentvalAcc:.4f}')
        print(f'Test - Acc: {currenttestAcc:.4f}')
        print(f'Best - Epoch: {BestEpoch}, Val_Acc: {BestAcc:.4f}, Test_Acc: {BestTestAcc:.4f}')
        print('=' * 60)

    print('\nFinal Results:')
    print(f'Method: base')
    print(f'BestValAcc: {BestAcc:.4f}')
    print(f'BestTestAcc: {BestTestAcc:.4f}')
