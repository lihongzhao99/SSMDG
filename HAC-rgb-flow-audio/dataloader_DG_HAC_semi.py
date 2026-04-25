from mmaction.datasets.pipelines import Compose
import torch.utils.data
import csv
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio
import torch
import torchvision.transforms.functional as F
import librosa
import random


def apply_ops(img, op_name, magnitude, H, W):
    """Apply one RandAugment-style image operation."""
    if op_name in ['AutoContrast', 'Equalize', 'Posterize', 'Solarize']:
        if img.dtype == torch.float32:
            img_uint8 = (img * 255).to(torch.uint8)
        else:
            img_uint8 = img

        if op_name == 'AutoContrast':
            return F.autocontrast(img_uint8).to(torch.float32) / 255.0
        elif op_name == 'Equalize':
            return F.equalize(img_uint8).to(torch.float32) / 255.0
        elif op_name == 'Posterize':
            posterize_bits = int(magnitude * 4) + 4
            return F.posterize(img_uint8, posterize_bits).to(torch.float32) / 255.0
        elif op_name == 'Solarize':
            solarize_threshold = 256.0 - magnitude * 255.0
            return F.solarize(img_uint8, solarize_threshold).to(torch.float32) / 255.0

    elif op_name == 'Color':
        return F.adjust_saturation(img, 1 + magnitude * 0.9)
    elif op_name == 'Contrast':
        return F.adjust_contrast(img, 1 + magnitude * 0.9)
    elif op_name == 'Brightness':
        return F.adjust_brightness(img, 1 + magnitude * 0.9)
    elif op_name == 'Sharpness':
        return F.adjust_sharpness(img, 1 + magnitude * 0.9)
    elif op_name == 'Rotate':
        degrees = magnitude * 30
        return F.rotate(img, degrees)
    elif op_name == 'ShearX':
        shear_angle = magnitude * 30
        return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear_angle, 0])
    elif op_name == 'ShearY':
        shear_angle = magnitude * 30
        return F.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[0, shear_angle])
    elif op_name == 'TranslateX':
        translate_px = magnitude * 0.4 * W
        return F.affine(img, angle=0, translate=[translate_px, 0], scale=1.0, shear=[0, 0])
    elif op_name == 'TranslateY':
        translate_px = magnitude * 0.4 * H
        return F.affine(img, angle=0, translate=[0, translate_px], scale=1.0, shear=[0, 0])
    else:
        return img


class UnifiedHACDomainSemiDataset(torch.utils.data.Dataset):
    def __init__(self, split='test', source=True, domain=['human'], modality='rgb', cfg=None, cfg_flow=None,
                 use_video=True, use_flow=False, use_audio=True, datapath='./data/',
                 is_labeled=True, semi_setting='number', semi_value=10, unlabeled_domains=None):
        """
        Unified semi-supervised dataset for HAC domain adaptation
        """
        self.base_path = datapath
        self.video_list = []
        self.prefix_list = []
        self.label_list = []
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow
        self.split = split
        self.domain = domain
        self.modality = modality
        self.is_labeled = is_labeled
        self.semi_setting = semi_setting
        self.semi_value = semi_value
        self.unlabeled_domains = unlabeled_domains if unlabeled_domains is not None else []
        self.interval = 9

        if semi_setting not in ['number', 'ratio', 'domain']:
            raise ValueError("semi_setting must be one of: 'number', 'ratio', 'domain'")

        all_data = []
        for dom in domain:
            csv_file = self.base_path + "HAC_Splits/HAC_%s_only_%s.csv" % (split, dom)

            with open(csv_file) as f:
                f_csv = csv.reader(f)
                domain_data = []
                for i, row in enumerate(f_csv):
                    domain_data.append((row[0], row[1], dom))

            if split == 'train':
                selected_data = self._select_data_by_setting(domain_data, dom)
            else:
                selected_data = domain_data

            all_data.extend(selected_data)

            if split == 'test' and not source:
                with open(self.base_path + "HAC_Splits/HAC_train_only_%s.csv" % (dom)) as f:
                    f_csv = csv.reader(f)
                    for i, row in enumerate(f_csv):
                        all_data.append((row[0], row[1], dom))

        for video, label, dom in all_data:
            self.video_list.append(video)
            self.label_list.append(label)
            self.prefix_list.append(dom + '/')

        self.samples = [(self.video_list[i], self.label_list[i], self.prefix_list[i].rstrip('/'))
                        for i in range(len(self.video_list))]

        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
            self.train = True
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)
            self.train = False

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.video_path_base = self.base_path + 'HAC/'
        if not os.path.exists(self.video_path_base):
            os.mkdir(self.video_path_base)

        self.ops = [
            'AutoContrast', 'Equalize', 'Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY'
        ]

    def _select_data_by_setting(self, domain_data, dom):
        """Select labeled/unlabeled data based on the semi-supervised setting"""
        if self.semi_setting == 'number':
            return self._select_by_number(domain_data, dom)
        elif self.semi_setting == 'ratio':
            return self._select_by_ratio(domain_data, dom)
        elif self.semi_setting == 'domain':
            return self._select_by_domain(domain_data, dom)
        else:
            raise ValueError(f"Unknown semi_setting: {self.semi_setting}")

    def _select_by_number(self, domain_data, dom):
        """Select data based on fixed number per class"""
        from collections import defaultdict
        grouped = defaultdict(list)
        for video, label, domain in domain_data:
            grouped[label].append((video, label, domain))

        if self.is_labeled:
            labeled_samples = []
            for label, samples in grouped.items():
                if len(samples) >= self.semi_value:
                    sampled = random.sample(samples, self.semi_value)
                else:
                    sampled = samples
                labeled_samples.extend(sampled)
                print(f"Domain {dom}, Class {label}: {len(sampled)} labeled samples selected")
            print(f"Domain {dom}: Total labeled samples: {len(labeled_samples)}")
            return labeled_samples
        else:
            labeled_indices = set()
            for label, samples in grouped.items():
                if len(samples) >= self.semi_value:
                    sampled = random.sample(samples, self.semi_value)
                else:
                    sampled = samples
                for s in sampled:
                    labeled_indices.add((s[0], s[1], s[2]))

            unlabeled_samples = [s for s in domain_data if s not in labeled_indices]
            print(f"Domain {dom}: Total unlabeled samples: {len(unlabeled_samples)}")
            return unlabeled_samples

    def _select_by_ratio(self, domain_data, dom):
        """Select data based on percentage"""
        total_samples = len(domain_data)
        num_labeled_samples = int(total_samples * self.semi_value)

        print(f"Domain {dom}: Total samples: {total_samples}, "
              f"Labeled samples: {num_labeled_samples} ({self.semi_value * 100:.1f}%), "
              f"Unlabeled samples: {total_samples - num_labeled_samples}")

        from collections import defaultdict
        grouped = defaultdict(list)
        for video, label, domain in domain_data:
            grouped[label].append((video, label, domain))

        if self.is_labeled:
            labeled_samples = []
            for label, samples in grouped.items():
                class_labeled_count = max(1, int(len(samples) * self.semi_value))
                class_labeled_count = min(class_labeled_count, len(samples))
                if len(samples) >= class_labeled_count:
                    sampled = random.sample(samples, class_labeled_count)
                else:
                    sampled = samples
                labeled_samples.extend(sampled)
                print(f"  Class {label}: {len(samples)} total -> {len(sampled)} labeled")
            print(f"  Final labeled dataset size: {len(labeled_samples)}")
            return labeled_samples
        else:
            labeled_indices = set()
            for label, samples in grouped.items():
                class_labeled_count = max(1, int(len(samples) * self.semi_value))
                class_labeled_count = min(class_labeled_count, len(samples))
                if len(samples) >= class_labeled_count:
                    sampled = random.sample(samples, class_labeled_count)
                else:
                    sampled = samples
                for s in sampled:
                    labeled_indices.add((s[0], s[1], s[2]))

            unlabeled_samples = [s for s in domain_data if s not in labeled_indices]
            print(f"  Final unlabeled dataset size: {len(unlabeled_samples)}")
            return unlabeled_samples

    def _select_by_domain(self, domain_data, dom):
        """Select data based on domain labeling"""
        if dom in self.unlabeled_domains:
            if self.is_labeled:
                print(f"Domain {dom}: All {len(domain_data)} samples are unlabeled, returning empty labeled set")
                return []
            else:
                print(f"Domain {dom}: All {len(domain_data)} samples are unlabeled")
                return domain_data
        else:
            if self.is_labeled:
                print(f"Domain {dom}: All {len(domain_data)} samples are labeled")
                return domain_data
            else:
                print(f"Domain {dom}: All {len(domain_data)} samples are labeled, returning empty unlabeled set")
                return []

    def apply_weak_video_augmentation(self, imgs):
        """Apply weak video augmentation"""
        if torch.rand(1) < 0.5:
            imgs = torch.flip(imgs, dims=[-1])

        H, W = imgs.shape[-2:]
        max_dx = int(W * 0.1)
        max_dy = int(H * 0.1)
        tx = torch.randint(-max_dx, max_dx + 1, (1,)).item()
        ty = torch.randint(-max_dy, max_dy + 1, (1,)).item()

        imgs_permuted = imgs.squeeze(0).permute(1, 0, 2, 3)
        augmented_frames = []
        for frame in imgs_permuted:
            aug_frame = F.affine(frame, angle=0, translate=(tx, ty), scale=1.0, shear=[0.0, 0.0], fill=0)
            augmented_frames.append(aug_frame)

        augmented_tensor = torch.stack(augmented_frames, dim=0)
        imgs = augmented_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        return imgs

    def apply_strong_video_augmentation(self, imgs):
        """Apply strong video augmentation"""
        N = random.randint(1, 2)
        M = random.uniform(0.05, 0.2)
        H, W = imgs.shape[-2:]

        op_list = random.sample(self.ops, k=N)
        imgs_permuted = imgs.squeeze(0).permute(1, 0, 2, 3)

        augmented_frames = []
        for frame in imgs_permuted:
            for op in op_list:
                frame = apply_ops(frame, op, M, H, W)
            augmented_frames.append(frame)

        imgs = torch.stack(augmented_frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)

        cutout_size = int(0.15 * H)
        cutout_x = random.randint(0, H - cutout_size)
        cutout_y = random.randint(0, W - cutout_size)
        imgs[:, :, :, cutout_y:cutout_y + cutout_size, cutout_x:cutout_x + cutout_size] = 0

        return imgs

    def apply_weak_flow_augmentation(self, flows):
        """Weak flow augmentation: random flip + translation."""
        if torch.rand(1) < 0.5:
            flows = torch.flip(flows, dims=[-1])

        H, W = flows.shape[-2:]
        max_dx = int(W * 0.1)
        max_dy = int(H * 0.1)
        tx = torch.randint(-max_dx, max_dx + 1, (1,)).item()
        ty = torch.randint(-max_dy, max_dy + 1, (1,)).item()

        flows_permuted = flows.squeeze(0).permute(1, 0, 2, 3)
        augmented_frames = []
        for frame in flows_permuted:
            aug_frame = F.affine(frame, angle=0, translate=(tx, ty), scale=1.0, shear=[0.0, 0.0], fill=0)
            augmented_frames.append(aug_frame)

        augmented_tensor = torch.stack(augmented_frames, dim=0)
        flows = augmented_tensor.permute(1, 0, 2, 3).unsqueeze(0)
        return flows

    def apply_strong_flow_augmentation(self, flows):
        """Strong flow augmentation: weak aug + Cutout + noise injection."""
        flows = self.apply_weak_flow_augmentation(flows)
        H, W = flows.shape[-2:]

        cutout_size = int(0.15 * H)
        cutout_x = random.randint(0, H - cutout_size)
        cutout_y = random.randint(0, W - cutout_size)
        flows[:, :, :, cutout_y:cutout_y + cutout_size, cutout_x:cutout_x + cutout_size] = 0

        noise_std = random.uniform(0.02, 0.08)
        flows = flows + torch.randn_like(flows) * noise_std
        return flows

    def apply_weak_audio_augmentation(self, samples, samplerate):
        """Apply weak audio augmentation"""
        gain = np.random.uniform(0.8, 1.2)
        samples = samples * gain
        n_steps = np.random.uniform(-1, 1)
        samples = librosa.effects.pitch_shift(samples, sr=samplerate, n_steps=n_steps, bins_per_octave=12)
        return samples

    def apply_strong_audio_augmentation(self, spectrogram):
        """Apply strong audio augmentation"""
        num_freq_bins = spectrogram.shape[0]
        freq_interval = min(self.interval * 2, num_freq_bins // 3)
        if num_freq_bins > freq_interval:
            start_freq = np.random.choice(num_freq_bins - freq_interval, (1,))[0]
            spectrogram[start_freq:(start_freq + freq_interval), :] = 0

        time_masking_interval = 40
        num_time_bins = spectrogram.shape[1]
        if num_time_bins > time_masking_interval:
            start_time = np.random.choice(num_time_bins - time_masking_interval, (1,))[0]
            spectrogram[:, start_time:(start_time + time_masking_interval)] = 0

        noise = np.random.uniform(-0.1, 0.1, spectrogram.shape)
        spectrogram = spectrogram + noise
        return spectrogram

    def __getitem__(self, index):
        label1 = int(self.label_list[index])
        video_path = self.video_path_base + self.video_list[index] + "/"
        video_path = video_path + self.video_list[index] + '-'

        if self.use_video:
            video_file = self.base_path + 'HAC_Splits/' + self.prefix_list[index] + 'videos/' + self.video_list[index]
            vid = iio.imread(video_file, plugin="pyav")
            frame_num = vid.shape[0]
            start_frame = 0
            end_frame = frame_num - 1

            filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
            modality = self.cfg.data.val.get('modality', 'RGB')
            start_index = self.cfg.data.val.get('start_index', start_frame)
            data = dict(
                frame_dir=video_path,
                total_frames=end_frame - start_frame,
                label=-1,
                start_index=start_index,
                video=vid,
                frame_num=frame_num,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data, frame_inds = self.pipeline(data)

            if self.split == 'train':
                imgs_weak = self.apply_weak_video_augmentation(data['imgs'].clone())

                if not self.is_labeled:
                    imgs_strong = self.apply_strong_video_augmentation(data['imgs'].clone())
                    data['imgs_strong'] = imgs_strong

                data['imgs'] = imgs_weak

        if self.use_flow:
            video_file_x = self.base_path + 'HAC_Splits/' + self.prefix_list[index] + 'flow/' + self.video_list[index][:-4] + '_flow_x.mp4'
            video_file_y = self.base_path + 'HAC_Splits/' + self.prefix_list[index] + 'flow/' + self.video_list[index][:-4] + '_flow_y.mp4'
            vid_x = iio.imread(video_file_x, plugin="pyav")
            vid_y = iio.imread(video_file_y, plugin="pyav")

            frame_num_flow = vid_x.shape[0]
            start_frame_flow = 0
            end_frame_flow = frame_num_flow - 1

            filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
            modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
            start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame_flow)
            flow = dict(
                frame_dir=video_path,
                total_frames=end_frame_flow - start_frame_flow,
                label=-1,
                start_index=start_index_flow,
                video=vid_x,
                video_y=vid_y,
                frame_num=frame_num_flow,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow)
            flow, frame_inds_flow = self.pipeline_flow(flow)

            if self.split == 'train':
                flow_weak = self.apply_weak_flow_augmentation(flow['imgs'].clone())
                if not self.is_labeled:
                    flow_strong = self.apply_strong_flow_augmentation(flow['imgs'].clone())
                    flow['imgs_strong'] = flow_strong
                flow['imgs'] = flow_weak

        if self.use_audio:
            audio_path = self.base_path + 'HAC_Splits/' + self.prefix_list[index] + 'audio/' + self.video_list[index][
                                                                                               :-4] + '.wav'
            if self.use_video:
                start_time = frame_inds[0] / 24.0
                end_time = frame_inds[-1] / 24.0
            else:
                start_time = frame_inds_flow[0] / 24.0
                end_time = frame_inds_flow[-1] / 24.0
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate

            start1 = start_time / duration * len(samples)
            end1 = end_time / duration * len(samples)
            start1 = int(np.round(start1))
            end1 = int(np.round(end1))
            samples = samples[start1:end1]

            if self.split == 'train':
                samples_weak = self.apply_weak_audio_augmentation(samples.copy(), samplerate)
            else:
                samples_weak = samples

            resamples_weak = samples_weak[:160000]
            if len(resamples_weak) == 0:
                resamples_weak = np.zeros((160000))
            while len(resamples_weak) < 160000:
                resamples_weak = np.tile(resamples_weak, 10)[:160000]

            resamples_weak[resamples_weak > 1.] = 1.
            resamples_weak[resamples_weak < -1.] = -1.
            _, _, spectrogram_weak = signal.spectrogram(resamples_weak, samplerate, nperseg=512, noverlap=353)
            spectrogram_weak = np.log(spectrogram_weak + 1e-7)

            mean = np.mean(spectrogram_weak)
            std = np.std(spectrogram_weak)
            spectrogram_weak = np.divide(spectrogram_weak - mean, std + 1e-9)

            if self.split == 'train':
                noise = np.random.uniform(-0.03, 0.03, spectrogram_weak.shape)
                spectrogram_weak = spectrogram_weak + noise

                if not self.is_labeled:
                    spectrogram_strong = self.apply_strong_audio_augmentation(spectrogram_weak.copy())

                    if self.use_video and self.use_flow and self.use_audio:
                        return (data, flow, spectrogram_weak.astype(np.float32),
                                spectrogram_strong.astype(np.float32),
                                label1, self._get_domain_label(index))

                    if self.use_video and self.use_audio:
                        return (data, spectrogram_weak.astype(np.float32),
                                spectrogram_strong.astype(np.float32),
                                label1, self._get_domain_label(index))

            if self.use_video and self.use_flow and self.use_audio:
                return data, flow, spectrogram_weak.astype(np.float32), label1, self._get_domain_label(index)
            if self.use_video and self.use_audio:
                return data, spectrogram_weak.astype(np.float32), label1, self._get_domain_label(index)
        else:
            print("INPUT ERROR: Both video and audio must be enabled.")
            return None, None, None, None

    def _get_domain_label(self, index):
        """Get domain label tensor"""
        current_domain = self.prefix_list[index].rstrip('/')
        if len(self.domain) != 2:
            return torch.tensor(-1)
        domain_mapping = {domain: domain_id for domain_id, domain in enumerate(self.domain)}
        mapped_value = domain_mapping.get(current_domain, -1)
        return torch.tensor(mapped_value)

    def __len__(self):
        return len(self.video_list)


class UnifiedSemiDataLoader:
    """
    Unified DataLoader wrapper for paired labeled and unlabeled batches
    """

    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size=16, num_workers=4):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

        labeled_size = len(labeled_dataset)
        unlabeled_size = len(unlabeled_dataset)

        print(f"Labeled dataset size: {labeled_size}")
        print(f"Unlabeled dataset size: {unlabeled_size}")

        if labeled_size == 0 or unlabeled_size == 0:
            raise ValueError("Both labeled and unlabeled datasets must have samples for semi-supervised training")

        self.labeled_loader = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        self.unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        self.batches_per_epoch = max(len(self.labeled_loader), len(self.unlabeled_loader))

        print(f"Labeled batches: {len(self.labeled_loader)}")
        print(f"Unlabeled batches: {len(self.unlabeled_loader)}")
        print(f"Batches per epoch: {self.batches_per_epoch}")

    def __iter__(self):
        self.labeled_iter = iter(self.labeled_loader)
        self.unlabeled_iter = iter(self.unlabeled_loader)
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.batches_per_epoch:
            raise StopIteration

        try:
            labeled_batch = next(self.labeled_iter)
        except StopIteration:
            self.labeled_iter = iter(self.labeled_loader)
            labeled_batch = next(self.labeled_iter)

        try:
            unlabeled_batch = next(self.unlabeled_iter)
        except StopIteration:
            self.unlabeled_iter = iter(self.unlabeled_loader)
            unlabeled_batch = next(self.unlabeled_iter)

        self.current_batch += 1
        return labeled_batch, unlabeled_batch

    def __len__(self):
        return self.batches_per_epoch


def create_unified_datasets(cfg, domains, datapath, use_video, use_audio,
                            semi_setting, semi_value, unlabeled_domains=None,
                            cfg_flow=None, use_flow=False):
    """
    Helper function to create labeled and unlabeled datasets based on semi-supervised setting
    """

    labeled_dataset = UnifiedHACDomainSemiDataset(
        split='train', source=True, domain=domains, cfg=cfg, cfg_flow=cfg_flow,
        datapath=datapath, use_video=use_video, use_flow=use_flow, use_audio=use_audio,
        is_labeled=True, semi_setting=semi_setting, semi_value=semi_value,
        unlabeled_domains=unlabeled_domains
    )

    unlabeled_dataset = UnifiedHACDomainSemiDataset(
        split='train', source=True, domain=domains, cfg=cfg, cfg_flow=cfg_flow,
        datapath=datapath, use_video=use_video, use_flow=use_flow, use_audio=use_audio,
        is_labeled=False, semi_setting=semi_setting, semi_value=semi_value,
        unlabeled_domains=unlabeled_domains
    )

    return labeled_dataset, unlabeled_dataset
