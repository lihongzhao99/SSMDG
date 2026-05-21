import os
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import copy
import random

DEFAULT_NUM_CLASSES = 7


def _save_if_present(state_dict_obj, state_key, models, model_key):
    module = models.get(model_key)
    if module is not None:
        state_dict_obj[state_key] = module.state_dict()


def _add_model_state_dicts(state_dict_obj, models):
    """Add all available model state dicts to a checkpoint."""
    if models['model'] is not None:
        state_dict_obj['video_model'] = models['model'].state_dict()
    if models.get('model_flow') is not None:
        state_dict_obj['flow_model'] = models['model_flow'].state_dict()
    if models['audio_model'] is not None:
        state_dict_obj['audio_model'] = models['audio_model'].state_dict()
    if models['audio_cls_model'] is not None:
        state_dict_obj['audio_cls_model'] = models['audio_cls_model'].state_dict()

    state_dict_obj['mlp_cls'] = models['mlp_cls'].state_dict()
    _save_if_present(state_dict_obj, 'mlp_v2a', models, 'mlp_v2a')
    _save_if_present(state_dict_obj, 'mlp_a2v', models, 'mlp_a2v')
    _save_if_present(state_dict_obj, 'mlp_v2f', models, 'mlp_v2f')
    _save_if_present(state_dict_obj, 'mlp_f2v', models, 'mlp_f2v')
    _save_if_present(state_dict_obj, 'mlp_f2a', models, 'mlp_f2a')
    _save_if_present(state_dict_obj, 'mlp_a2f', models, 'mlp_a2f')
    return state_dict_obj


def save_best_model(epoch, models, optim, best_acc, best_loss,
                    checkpoint_dir="checkpoints"):
    """Save the best validation checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'best_acc': best_acc,
        'best_loss': best_loss,
        'optimizer_state': optim.state_dict(),
    }
    _add_model_state_dicts(checkpoint, models)
    save_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    print(f"Best model saved to: {save_path}")
    return save_path


class PCGrad:
    """
    Gradient Surgery for Multi-Task Learning
    """
    def __init__(self, optimizer, reduction='mean'):
        self.optimizer = optimizer
        self.reduction = reduction

    def _pack_grad(self, objectives, scaler=None):
        grads = []
        for obj in objectives:
            self.optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(obj).backward(retain_graph=True)
            else:
                obj.backward(retain_graph=True)

            grad = []
            for p in self.optimizer.param_groups[0]['params']:
                if p.grad is None:
                    grad.append(torch.zeros_like(p).to(p.device))
                else:
                    grad.append(p.grad.clone())
            grads.append(grad)
        return grads

    def _project_conflicting(self, grads):
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for i in range(num_task):
            task_index = list(range(num_task))
            random.shuffle(task_index)
            for j in task_index:
                if j == i:
                    continue
                g_ij = sum([torch.dot(pc_grad[i][k].flatten(), grads[j][k].flatten())
                            for k in range(len(grads[i]))])
                if g_ij < 0:
                    g_j_norm_sq = sum([torch.sum(grads[j][k] ** 2)
                                       for k in range(len(grads[j]))])
                    for k in range(len(pc_grad[i])):
                        pc_grad[i][k] -= (g_ij / (g_j_norm_sq + 1e-8)) * grads[j][k]

        merged_grad = []
        for i in range(len(pc_grad[0])):
            if self.reduction == 'mean':
                merged_grad.append(sum([pc_grad[j][i] for j in range(num_task)]) / num_task)
            elif self.reduction == 'sum':
                merged_grad.append(sum([pc_grad[j][i] for j in range(num_task)]))
            else:
                raise ValueError(f'Unknown reduction: {self.reduction}')

        return merged_grad

    def step(self, objectives, scaler=None):
        grads = self._pack_grad(objectives, scaler=scaler)
        pc_grad = self._project_conflicting(grads)

        self.optimizer.zero_grad(set_to_none=True)
        for idx, p in enumerate(self.optimizer.param_groups[0]['params']):
            p.grad = pc_grad[idx]
        if scaler is not None:
            scaler.unscale_(self.optimizer)
            scaler.step(self.optimizer)
            scaler.update()
        else:
            self.optimizer.step()


class PrototypeManager:
    """
    Manage modality-class-domain prototypes.
    Supports different feature dimensions per modality.
    """

    def __init__(self, num_domains, num_classes, modality_dims, device):
        """
        Args:
            num_domains: Number of domains
            num_classes: Number of classes
            modality_dims: Modality feature dimensions, such as {'video': 2304, 'audio': 512}
            device: Target device
        """
        self.num_domains = num_domains
        self.num_classes = num_classes
        self.modality_dims = modality_dims  # {'video': 2304, 'audio': 512}
        self.modality_names = list(modality_dims.keys())
        self.device = device

        # Initialize prototypes for each modality.
        self.prototypes = {}
        self.prototype_counts = {}

        for modality, feature_dim in modality_dims.items():
            # [domains, classes, feature_dim]
            self.prototypes[modality] = F.normalize(
                torch.randn(num_domains, num_classes, feature_dim, device=device),
                dim=-1
            )
            # [domains, classes]
            self.prototype_counts[modality] = torch.zeros(
                num_domains, num_classes, device=device
            )

        # Prototype update rate.
        self.prototype_lr = 0.1

    def update_prototypes_from_batch(self, features_dict, labels, domain_labels, mask=None):
        """
        Update prototypes from labeled batch averages.

        Args:
            features_dict: {'video': video_features, 'audio': audio_features}
                          shape: [batch_size, feature_dim]
            labels: Class labels [batch_size]
            domain_labels: Domain labels [batch_size]
            mask: Valid sample mask [batch_size]; None means all samples are valid
        """
        if mask is None:
            mask = torch.ones(labels.size(0), dtype=torch.bool, device=self.device)

        # Average features for each (domain, class, modality) group.
        for domain in range(self.num_domains):
            for class_id in range(self.num_classes):
                # Select samples from the current domain and class.
                domain_class_mask = (domain_labels == domain) & (labels == class_id) & mask

                if domain_class_mask.sum() > 0:
                    for modality in self.modality_names:
                        if modality not in features_dict:
                            continue

                        # Compute the group mean feature.
                        features = F.normalize(features_dict[modality], dim=-1)
                        batch_features = features[domain_class_mask]  # [num_samples, feature_dim]
                        mean_feature = batch_features.mean(dim=0)  # [feature_dim]
                        mean_feature = F.normalize(mean_feature, dim=-1)

                        # Update the prototype with EMA-style averaging.
                        old_prototype = self.prototypes[modality][domain, class_id]
                        count = self.prototype_counts[modality][domain, class_id]

                        # EMA: p_new = (p_old * count + mean_feature) / (count + 1)
                        new_prototype = (old_prototype * count + mean_feature) / (count + 1)
                        new_prototype = F.normalize(new_prototype, dim=-1)

                        self.prototypes[modality][domain, class_id] = new_prototype
                        self.prototype_counts[modality][domain, class_id] += 1

    def get_mse_prototype_loss(self, features_dict, labels, domain_labels, mask=None):
        """
        Compute the MSE prototype alignment loss.

        Strategy:
        (1) Align each modality feature with the same-domain class prototype.
        (2) Align each modality feature with other-domain class prototypes.

        Args:
            features_dict: {'video': video_features, 'audio': audio_features}
            labels: Class labels [batch_size]
            domain_labels: Domain labels [batch_size]
            mask: Valid sample mask [batch_size]

        Returns:
            MSE prototype loss
        """
        if mask is None:
            mask = torch.ones(labels.size(0), dtype=torch.bool, device=self.device)

        valid_count = mask.sum().item()
        if valid_count == 0:
            return torch.tensor(0.0, device=self.device)

        # Collect valid indices.
        valid_indices = torch.where(mask)[0]

        total_loss = 0.0
        loss_count = 0

        for modality in self.modality_names:
            if modality not in features_dict:
                continue

            features = F.normalize(features_dict[modality], dim=-1)

            for idx in valid_indices:
                label = labels[idx].item()
                domain = domain_labels[idx].item()
                feature = features[idx]  # [feature_dim]

                # Same-domain class prototype.
                same_domain_proto = self.prototypes[modality][domain, label]
                same_domain_mse = torch.norm(feature - same_domain_proto, p=2)
                total_loss += same_domain_mse
                loss_count += 1

                # Average over other-domain class prototypes.
                for other_domain in range(self.num_domains):
                    if other_domain != domain:
                        other_domain_proto = self.prototypes[modality][other_domain, label]
                        other_domain_mse = torch.norm(feature - other_domain_proto, p=2)
                        total_loss += other_domain_mse
                        loss_count += 1

        mse_loss = total_loss / loss_count if loss_count > 0 else torch.tensor(0.0, device=self.device)
        return mse_loss

    def get_prototype_by_class_domain(self, domain, class_id, modality='video'):
        """Get a prototype by modality, domain, and class"""
        return self.prototypes[modality][domain, class_id]


def setup_prototype_manager(args, num_domains=2):
    """
    Initialize the prototype manager.

    Args:
        args: Training arguments
        num_domains: Number of domains

    Returns:
        prototype_manager
    """
    num_classes = args.num_classes

    # Set feature dimensions for enabled modalities.
    modality_dims = {}
    if args.use_video:
        modality_dims['video'] = 2304
    if args.use_flow:
        modality_dims['flow'] = 2048
    if args.use_audio:
        modality_dims['audio'] = 512

    prototype_manager = PrototypeManager(
        num_domains=num_domains,
        num_classes=num_classes,
        modality_dims=modality_dims,
        device='cuda:0'
    )

    return prototype_manager



def calculate_per_class_metrics(predictions, labels, num_classes=DEFAULT_NUM_CLASSES):
    """Calculate per-class sample counts and accuracy"""
    class_stats = {}

    for class_id in range(num_classes):
        class_mask = (labels == class_id)
        class_count = class_mask.sum().item()

        if class_count > 0:
            class_predictions = predictions[class_mask]
            class_correct = (class_predictions == class_id).sum().item()
            class_accuracy = class_correct / class_count
        else:
            class_accuracy = 0.0

        class_stats[class_id] = {
            'count': class_count,
            'accuracy': class_accuracy,
            'correct': class_correct if class_count > 0 else 0
        }

    return class_stats


def calculate_pseudo_label_per_class_stats(pseudo_labels, true_labels, mask, num_classes=DEFAULT_NUM_CLASSES):
    """
    Calculate per-class pseudo-label statistics
    """
    class_stats = {}

    selected_mask = mask > 0
    selected_pseudo = pseudo_labels[selected_mask]
    selected_true = true_labels[selected_mask]

    for class_id in range(num_classes):
        class_mask = (selected_pseudo == class_id)
        class_count = class_mask.sum().item()

        if class_count > 0:
            class_true_labels = selected_true[class_mask]
            class_correct = (class_true_labels == class_id).sum().item()
            class_accuracy = class_correct / class_count
        else:
            class_correct = 0
            class_accuracy = 0.0

        class_stats[class_id] = {
            'count': class_count,
            'accuracy': class_accuracy,
            'correct': class_correct
        }

    return class_stats


def calculate_pseudo_label_domain_stats(domain_labels, mask):
    """
    Calculate domain distribution of pseudo-labels
    """
    domain_stats = {}

    selected_mask = mask > 0
    selected_domains = domain_labels[selected_mask]

    unique_domains = torch.unique(selected_domains)

    for domain_id in unique_domains:
        domain_id_item = domain_id.item()
        domain_mask = (selected_domains == domain_id)
        domain_count = domain_mask.sum().item()

        domain_stats[domain_id_item] = {
            'count': domain_count,
            'percentage': 0.0
        }

    total_selected = selected_mask.sum().item()
    if total_selected > 0:
        for domain_id in domain_stats:
            domain_stats[domain_id]['percentage'] = domain_stats[domain_id]['count'] / total_selected * 100

    return domain_stats


def get_cross_modal_translation_loss(features_dict, labels, domain_labels,
                                     prototype_manager, translation_heads,
                                     args, mask=None):
    """
    Compute cross-modal translation loss.

    Args:
        features_dict: {'video': video_features, 'audio': audio_features}
                      shape: [batch_size, feature_dim]
        labels: Class labels [batch_size]
        domain_labels: Domain labels [batch_size]
        prototype_manager: Prototype manager
        translation_heads: Modality translation heads
        args: Runtime arguments
        mask: Valid sample mask [batch_size]; None means all samples are valid

    Returns:
        cross_modal_loss: Cross-modal translation loss
    """
    if mask is None:
        feat0 = next(iter(features_dict.values()))
        mask = torch.ones(labels.size(0), dtype=torch.bool, device=feat0.device)

    valid_count = mask.sum().item()
    if valid_count == 0:
        feat0 = next(iter(features_dict.values()))
        return torch.tensor(0.0, device=feat0.device)

    device = next(iter(features_dict.values())).device
    total_loss = 0.0
    loss_count = 0

    def _pair_translation_loss(src_mod, tgt_mod, head_key):
        nonlocal total_loss, loss_count
        if src_mod not in features_dict or tgt_mod not in features_dict:
            return
        if head_key not in translation_heads or translation_heads[head_key] is None:
            return

        src_feat = F.normalize(features_dict[src_mod], dim=-1)
        tgt_translated = translation_heads[head_key](src_feat)
        tgt_translated = F.normalize(tgt_translated, dim=-1)

        for idx in range(labels.size(0)):
            if not mask[idx]:
                continue

            label = labels[idx].item()
            domain = domain_labels[idx].item()

            proto_same_domain = prototype_manager.get_prototype_by_class_domain(
                domain, label, modality=tgt_mod
            )
            proto_same_domain = F.normalize(proto_same_domain, dim=-1)
            total_loss += torch.norm(tgt_translated[idx] - proto_same_domain, p=2)
            loss_count += 1

            distance_diff_domains = 0.0
            for other_domain in range(prototype_manager.num_domains):
                if other_domain != domain:
                    proto_diff_domain = prototype_manager.get_prototype_by_class_domain(
                        other_domain, label, modality=tgt_mod
                    )
                    proto_diff_domain = F.normalize(proto_diff_domain, dim=-1)
                    distance_diff_domains += torch.norm(tgt_translated[idx] - proto_diff_domain, p=2)

            if prototype_manager.num_domains > 1:
                distance_diff_domains /= (prototype_manager.num_domains - 1)
                total_loss += distance_diff_domains
                loss_count += 1

    _pair_translation_loss('video', 'audio', 'v2a')
    _pair_translation_loss('audio', 'video', 'a2v')
    _pair_translation_loss('video', 'flow', 'v2f')
    _pair_translation_loss('flow', 'video', 'f2v')
    _pair_translation_loss('flow', 'audio', 'f2a')
    _pair_translation_loss('audio', 'flow', 'a2f')

    cross_modal_loss = total_loss / loss_count if loss_count > 0 else torch.tensor(0.0, device=device)
    return cross_modal_loss


def _build_selected_features_dict(video_emd, flow_emd, audio_emd, args, mask=None):
    features_dict = {
        'video': video_emd if args.use_video else None,
        'flow': flow_emd if args.use_flow else None,
        'audio': audio_emd if args.use_audio else None
    }
    features_dict = {k: v for k, v in features_dict.items() if v is not None}

    if mask is not None:
        features_dict = {k: v[mask] for k, v in features_dict.items()}

    return features_dict


def _extract_multimodal_logits_and_embeddings(clip, flow, spectrogram, models, args):
    model = models['model']
    model_flow = models['model_flow']
    audio_model = models['audio_model']
    audio_cls_model = models['audio_cls_model']

    with torch.no_grad():
        if args.use_video:
            x_slow, x_fast = model.module.backbone.get_feature(clip)
            video_feat = (x_slow, x_fast)
        if args.use_flow:
            flow_feat = model_flow.module.backbone.get_feature(flow)
        if args.use_audio:
            _, audio_feat, _ = audio_model(spectrogram)

    v_predict, v_emd, f_predict, f_emd, a_predict, a_emd = None, None, None, None, None, None

    if args.use_video:
        video_feat = model.module.backbone.get_predict(video_feat)
        v_predict, v_emd = model.module.cls_head(video_feat)

    if args.use_flow:
        flow_feat = model_flow.module.backbone.get_predict(flow_feat)
        f_predict, f_emd = model_flow.module.cls_head(flow_feat)

    if args.use_audio:
        a_predict, a_emd = audio_cls_model(audio_feat)

    return v_predict, v_emd, f_predict, f_emd, a_predict, a_emd


def _compute_gce_consistency_loss(unlabeled_logits_w, unlabeled_logits_s, fusion_pred,
                                  low_conf_consistent_mask, args):
    if low_conf_consistent_mask.sum() == 0:
        return torch.tensor(0.0, device=unlabeled_logits_w['fusion'].device)

    gce_fusion_loss_w = generalized_cross_entropy(
        unlabeled_logits_w['fusion'][low_conf_consistent_mask],
        fusion_pred[low_conf_consistent_mask],
        q=0.7
    )
    gce_fusion_loss_s = generalized_cross_entropy(
        unlabeled_logits_s['fusion'][low_conf_consistent_mask],
        fusion_pred[low_conf_consistent_mask],
        q=0.7
    )
    gce_loss = gce_fusion_loss_w + gce_fusion_loss_s
    for modal_key in ['video', 'flow', 'audio']:
        if modal_key in unlabeled_logits_w and modal_key in unlabeled_logits_s:
            gce_loss = gce_loss + generalized_cross_entropy(
                unlabeled_logits_w[modal_key][low_conf_consistent_mask],
                fusion_pred[low_conf_consistent_mask],
                q=0.7
            )
            gce_loss = gce_loss + generalized_cross_entropy(
                unlabeled_logits_s[modal_key][low_conf_consistent_mask],
                fusion_pred[low_conf_consistent_mask],
                q=0.7
            )
    return args.lambda_gce * gce_loss


def _compute_pseudo_label_masks(unlabeled_predict_w, modality_logits_w, threshold):
    fusion_probs = F.softmax(unlabeled_predict_w, dim=1)
    fusion_max_probs, fusion_pred = torch.max(fusion_probs, dim=1)

    mask = torch.zeros_like(fusion_max_probs, dtype=torch.bool)
    for modal_key, modal_logits in modality_logits_w.items():
        modal_probs = F.softmax(modal_logits, dim=1)
        modal_max_probs, modal_pred = torch.max(modal_probs, dim=1)
        modal_consistent = (fusion_pred == modal_pred) & modal_max_probs.ge(threshold) & fusion_max_probs.ge(threshold)
        mask = mask | modal_consistent

    low_conf_consistent_mask = fusion_max_probs.ge(threshold) & ~mask

    return fusion_pred, mask, low_conf_consistent_mask

def train_one_step_base(labeled_batch, unlabeled_batch, _epoch, models, criterion, optim, args,
                        prototype_manager, threshold, scaler=None):
    mlp_cls = models['mlp_cls']
    translation_heads = {
        'v2a': models.get('mlp_v2a'),
        'a2v': models.get('mlp_a2v'),
        'v2f': models.get('mlp_v2f'),
        'f2v': models.get('mlp_f2v'),
        'f2a': models.get('mlp_f2a'),
        'a2f': models.get('mlp_a2f'),
    }
    if not hasattr(optim, 'pcgrad'):
        optim.pcgrad = PCGrad(optim, reduction='mean')

    if args.use_flow:
        labeled_clip, labeled_flow, labeled_spectrogram, labeled_labels, labeled_domain_labels = labeled_batch
    else:
        labeled_clip, labeled_spectrogram, labeled_labels, labeled_domain_labels = labeled_batch
    labeled_labels = labeled_labels.cuda()
    labeled_domain_labels = labeled_domain_labels.cuda()

    if args.use_flow:
        unlabeled_clip, unlabeled_flow, unlabeled_spectrogram_weak, unlabeled_spectrogram_strong, unlabeled_labels, unlabeled_domain_labels = unlabeled_batch
    else:
        unlabeled_clip, unlabeled_spectrogram_weak, unlabeled_spectrogram_strong, unlabeled_labels, unlabeled_domain_labels = unlabeled_batch
    unlabeled_labels_cuda = unlabeled_labels.cuda()
    unlabeled_domain_labels_cuda = unlabeled_domain_labels.cuda()

    batch_unlabeled_size = unlabeled_labels.size(0)

    if args.use_video:
        labeled_clip = labeled_clip['imgs'].cuda().squeeze(1)
        unlabeled_clip_weak = unlabeled_clip['imgs'].cuda().squeeze(1)
        unlabeled_clip_strong = unlabeled_clip['imgs_strong'].cuda().squeeze(1)
    if args.use_flow:
        labeled_flow = labeled_flow['imgs'].cuda().squeeze(1)
        unlabeled_flow_weak = unlabeled_flow['imgs'].cuda().squeeze(1)
        unlabeled_flow_strong = unlabeled_flow['imgs_strong'].cuda().squeeze(1)
    if args.use_audio:
        labeled_spectrogram = labeled_spectrogram.unsqueeze(1).cuda()
        unlabeled_spectrogram_weak = unlabeled_spectrogram_weak.unsqueeze(1).cuda()
        unlabeled_spectrogram_strong = unlabeled_spectrogram_strong.unsqueeze(1).cuda()

    with autocast(enabled=args.use_amp):
        # ---------- labeled ----------
        labeled_v_predict, labeled_v_emd, labeled_f_predict, labeled_f_emd, labeled_a_predict, labeled_audio_emd = \
            _extract_multimodal_logits_and_embeddings(labeled_clip, labeled_flow if args.use_flow else None,
                                                      labeled_spectrogram, models, args)

        labeled_emd_list = []
        if args.use_video:
            labeled_emd_list.append(labeled_v_emd)
        if args.use_flow:
            labeled_emd_list.append(labeled_f_emd)
        if args.use_audio:
            labeled_emd_list.append(labeled_audio_emd)
        labeled_feat = torch.cat(labeled_emd_list, dim=1)
        labeled_predict = mlp_cls(labeled_feat)

        labeled_fusion_loss = criterion(labeled_predict, labeled_labels)
        labeled_loss = labeled_fusion_loss
        if args.use_video and labeled_v_predict is not None:
            labeled_loss = labeled_loss + criterion(labeled_v_predict, labeled_labels)
        if args.use_flow and labeled_f_predict is not None:
            labeled_loss = labeled_loss + criterion(labeled_f_predict, labeled_labels)
        if args.use_audio and labeled_a_predict is not None:
            labeled_loss = labeled_loss + criterion(labeled_a_predict, labeled_labels)

        # ---------- Update prototypes with labeled data ----------
        with torch.no_grad():
            labeled_features_dict = _build_selected_features_dict(
                labeled_v_emd, labeled_f_emd, labeled_audio_emd, args, mask=None
            )

            prototype_manager.update_prototypes_from_batch(
                labeled_features_dict,
                labeled_labels,
                labeled_domain_labels,
                mask=None
            )

        # ---------- unlabeled weak ----------
        unlabeled_v_predict_w, unlabeled_v_emd_w, unlabeled_f_predict_w, unlabeled_f_emd_w, unlabeled_a_predict_w, unlabeled_a_emd_w = \
            _extract_multimodal_logits_and_embeddings(unlabeled_clip_weak,
                                                      unlabeled_flow_weak if args.use_flow else None,
                                                      unlabeled_spectrogram_weak, models, args)

        unlabeled_emd_w_list = []
        if args.use_video:
            unlabeled_emd_w_list.append(unlabeled_v_emd_w)
        if args.use_flow:
            unlabeled_emd_w_list.append(unlabeled_f_emd_w)
        if args.use_audio:
            unlabeled_emd_w_list.append(unlabeled_a_emd_w)
        unlabeled_feat_w = torch.cat(unlabeled_emd_w_list, dim=1)
        unlabeled_predict_w = mlp_cls(unlabeled_feat_w)

        # ---------- pseudo label ----------
        modality_logits_w = {}
        if args.use_video:
            modality_logits_w['video'] = unlabeled_v_predict_w
        if args.use_flow:
            modality_logits_w['flow'] = unlabeled_f_predict_w
        if args.use_audio:
            modality_logits_w['audio'] = unlabeled_a_predict_w
        pseudo_labels, mask, low_conf_consistent_mask = _compute_pseudo_label_masks(
            unlabeled_predict_w, modality_logits_w, threshold
        )
        initial_pseudo_count = mask.sum().item()

        # ---------- unlabeled strong ----------
        unlabeled_v_predict_s, unlabeled_v_emd_s, unlabeled_f_predict_s, unlabeled_f_emd_s, unlabeled_a_predict_s, unlabeled_a_emd_s = \
            _extract_multimodal_logits_and_embeddings(unlabeled_clip_strong,
                                                      unlabeled_flow_strong if args.use_flow else None,
                                                      unlabeled_spectrogram_strong, models, args)

        unlabeled_emd_s_list = []
        if args.use_video:
            unlabeled_emd_s_list.append(unlabeled_v_emd_s)
        if args.use_flow:
            unlabeled_emd_s_list.append(unlabeled_f_emd_s)
        if args.use_audio:
            unlabeled_emd_s_list.append(unlabeled_a_emd_s)
        unlabeled_feat_s = torch.cat(unlabeled_emd_s_list, dim=1)
        unlabeled_predict_s = mlp_cls(unlabeled_feat_s)

        # ---------- unsupervised loss ----------
        if mask.sum() > 0:
            unsup_loss = F.cross_entropy(unlabeled_predict_s[mask], pseudo_labels[mask])
            if args.use_video:
                unsup_loss = unsup_loss + F.cross_entropy(unlabeled_v_predict_s[mask], pseudo_labels[mask])
            if args.use_flow:
                unsup_loss = unsup_loss + F.cross_entropy(unlabeled_f_predict_s[mask], pseudo_labels[mask])
            if args.use_audio:
                unsup_loss = unsup_loss + F.cross_entropy(unlabeled_a_predict_s[mask], pseudo_labels[mask])
            unlabeled_loss = args.lambda_u * unsup_loss
        else:
            unlabeled_loss = torch.tensor(0.0).cuda()

        # ---------- GCE ----------
        unlabeled_logits_w = {'fusion': unlabeled_predict_w}
        if args.use_video:
            unlabeled_logits_w['video'] = unlabeled_v_predict_w
        if args.use_audio:
            unlabeled_logits_w['audio'] = unlabeled_a_predict_w
        if args.use_flow:
            unlabeled_logits_w['flow'] = unlabeled_f_predict_w
        unlabeled_logits_s = {'fusion': unlabeled_predict_s}
        if args.use_video:
            unlabeled_logits_s['video'] = unlabeled_v_predict_s
        if args.use_audio:
            unlabeled_logits_s['audio'] = unlabeled_a_predict_s
        if args.use_flow:
            unlabeled_logits_s['flow'] = unlabeled_f_predict_s
        unlabeled_loss += _compute_gce_consistency_loss(
            unlabeled_logits_w, unlabeled_logits_s, pseudo_labels, low_conf_consistent_mask, args
        )

        # ---------- MSE prototype alignment loss ----------
        if mask.sum() > 0 and args.lambda_mse_proto > 0:
            unlabeled_features_dict = _build_selected_features_dict(
                unlabeled_v_emd_s, unlabeled_f_emd_s, unlabeled_a_emd_s, args, mask=mask
            )

            mse_proto_loss = prototype_manager.get_mse_prototype_loss(
                unlabeled_features_dict,
                pseudo_labels[mask],
                unlabeled_domain_labels_cuda[mask],
                mask=None
            )
            unlabeled_loss += args.lambda_mse_proto * mse_proto_loss

        # ---------- Cross-modal translation loss ----------
        if mask.sum() > 0:
            unlabeled_features_dict_for_translation = _build_selected_features_dict(
                unlabeled_v_emd_s, unlabeled_f_emd_s, unlabeled_a_emd_s, args, mask=mask
            )
            unlabeled_features_dict_for_translation_w = _build_selected_features_dict(
                unlabeled_v_emd_w, unlabeled_f_emd_w, unlabeled_a_emd_w, args, mask=mask
            )

            cross_modal_loss_s = get_cross_modal_translation_loss(
                unlabeled_features_dict_for_translation,
                pseudo_labels[mask],
                unlabeled_domain_labels_cuda[mask],
                prototype_manager,
                translation_heads,
                args,
                mask=None
            )

            cross_modal_loss_w = get_cross_modal_translation_loss(
                unlabeled_features_dict_for_translation_w,
                pseudo_labels[mask],
                unlabeled_domain_labels_cuda[mask],
                prototype_manager,
                translation_heads,
                args,
                mask=None
            )
            cross_modal_loss = cross_modal_loss_s + cross_modal_loss_w
            unlabeled_loss += args.lambda_cross_modal * cross_modal_loss


    # ---------- metrics ----------
    selected_pseudo = pseudo_labels[mask]
    selected_true = unlabeled_labels_cuda[mask]
    pseudo_total = mask.sum().item()
    pseudo_correct = (selected_pseudo == selected_true).sum().item()
    pseudo_acc = pseudo_correct / pseudo_total if pseudo_total > 0 else 0.0

    per_class_stats = calculate_pseudo_label_per_class_stats(
        pseudo_labels, unlabeled_labels_cuda, mask, num_classes=args.num_classes
    )

    domain_stats = calculate_pseudo_label_domain_stats(
        unlabeled_domain_labels_cuda, mask
    )

    ce_count = mask.sum().item()
    gce_count = low_conf_consistent_mask.sum().item()

    # ---------- optimize ----------
    objectives = [labeled_loss, unlabeled_loss]
    objectives = [obj for obj in objectives if obj.item() > 0]
    if not objectives:
        optim.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=labeled_labels.device)
    elif len(objectives) > 1:
        optim.pcgrad.step(objectives, scaler=scaler if args.use_amp else None)
        total_loss = sum([obj.detach() for obj in objectives])
    else:
        optim.zero_grad(set_to_none=True)
        if args.use_amp and scaler is not None:
            scaler.scale(objectives[0]).backward()
            scaler.step(optim)
            scaler.update()
        else:
            objectives[0].backward()
            optim.step()
        total_loss = objectives[0].detach()


    return (
        labeled_predict, total_loss, labeled_loss, unlabeled_loss,
        pseudo_acc, pseudo_correct, pseudo_total, per_class_stats,
        domain_stats, initial_pseudo_count, ce_count, gce_count, batch_unlabeled_size
    )

def generalized_cross_entropy(pred, target, q=0.7):
    """
    Generalized Cross Entropy
    """
    probs = F.softmax(pred, dim=1)
    target_probs = probs.gather(1, target.unsqueeze(1)).squeeze()
    target_probs = torch.clamp(target_probs, min=1e-8, max=1.0)
    loss = (1 - torch.pow(target_probs, q)) / q
    return loss.mean()


def validate_one_step(clip, labels, flow, spectrogram, models, criterion, args):
    """Validation step"""
    model = models['model']
    model_flow = models['model_flow']
    audio_model = models['audio_model']
    audio_cls_model = models['audio_cls_model']
    mlp_cls = models['mlp_cls']

    labels = labels.cuda()
    if args.use_video:
        clip = clip['imgs'].cuda().squeeze(1)
    if args.use_flow:
        flow = flow['imgs'].cuda().squeeze(1)
    if args.use_audio:
        spectrogram = spectrogram.unsqueeze(1).type(torch.FloatTensor).cuda()

    with torch.no_grad():
        with autocast(enabled=args.use_amp):
            if args.use_video:
                x_slow, x_fast = model.module.backbone.get_feature(clip)
                v_feat = (x_slow.detach(), x_fast.detach())
                v_feat = model.module.backbone.get_predict(v_feat)
                v_predict, v_emd = model.module.cls_head(v_feat)

            if args.use_flow:
                f_feat = model_flow.module.backbone.get_feature(flow)
                f_feat = model_flow.module.backbone.get_predict(f_feat)
                f_predict, f_emd = model_flow.module.cls_head(f_feat)

            if args.use_audio:
                _, audio_feat, _ = audio_model(spectrogram)
                a_predict, audio_emd = audio_cls_model(audio_feat.detach())

            emd_list = []
            if args.use_video:
                emd_list.append(v_emd)
            if args.use_flow:
                emd_list.append(f_emd)
            if args.use_audio:
                emd_list.append(audio_emd)
            feat = torch.cat(emd_list, dim=1)

            predict = mlp_cls(feat)

    loss = criterion(predict, labels)
    return predict, loss



class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=DEFAULT_NUM_CLASSES, hidden=512):
        super(Encoder, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        return self.enc_net(feat)


class EncoderTrans(nn.Module):
    def __init__(self, input_dim=2816, out_dim=DEFAULT_NUM_CLASSES, hidden=512):
        super(EncoderTrans, self).__init__()
        self.enc_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, feat):
        feat = self.enc_net(feat)
        return feat

def get_log_name(args):
    """Generate log name based on semi-supervised setting"""
    log_name = f"base_{args.semi_setting}"

    if args.semi_setting == 'number':
        log_name += f"{args.semi_value}"
    elif args.semi_setting == 'ratio':
        log_name += f"{args.semi_value * 100:.1f}"
    elif args.semi_setting == 'domain':
        unlabeled_str = '_'.join(args.unlabeled_domains) if args.unlabeled_domains else 'none'
        log_name += f"_unlabeled{unlabeled_str}"

    log_name += f"_log{args.source_domain}2{args.target_domain}"

    if args.use_video:
        log_name += '_video'
    if args.use_flow:
        log_name += '_flow'
    if args.use_audio:
        log_name += '_audio'

    log_name += args.appen
    return log_name
