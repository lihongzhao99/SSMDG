import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import torch
import torchvision.transforms.functional as F
import librosa
from mmaction.datasets.pipelines import Compose
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


class UnifiedEPICDomainSemiDataset(torch.utils.data.Dataset):
    def __init__(self, split='train', domain=['D1'], modality='rgb', cfg=None, cfg_flow=None, sample_dur=10,
                 use_video=True, use_flow=False, use_audio=True, datapath='./data/',
                 is_labeled=True, semi_setting='number', semi_value=10, unlabeled_domains=None):
        """
        Unified semi-supervised dataset for EPIC-KITCHENS domain adaptation
        """
        self.base_path = datapath
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur
        self.use_video = use_video
        self.use_flow = use_flow
        self.use_audio = use_audio
        self.is_labeled = is_labeled
        self.semi_setting = semi_setting
        self.semi_value = semi_value
        self.unlabeled_domains = unlabeled_domains if unlabeled_domains is not None else []
        self.domain = domain

        # Validate parameters
        if semi_setting not in ['number', 'ratio', 'domain']:
            raise ValueError("semi_setting must be one of: 'number', 'ratio', 'domain'")

        if semi_setting == 'number' and not isinstance(semi_value, int):
            raise ValueError("For 'number' setting, semi_value must be an integer")

        if semi_setting == 'ratio' and (not isinstance(semi_value, (int, float)) or not 0.0 <= semi_value <= 1.0):
            raise ValueError("For 'ratio' setting, semi_value must be a float between 0.0 and 1.0")

        # Build the data pipeline.
        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
            if self.use_flow:
                train_pipeline_flow = cfg_flow.data.train.pipeline
                self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)
            if self.use_flow:
                val_pipeline_flow = cfg_flow.data.val.pipeline
                self.pipeline_flow = Compose(val_pipeline_flow)

        samples = []

        for dom in domain:
            train_file = pd.read_pickle(
                self.base_path + 'MM-SADA_Domain_Adaptation_Splits/' + dom + "_" + split + ".pkl")

            if split == 'train':
                selected_data = self._select_data_by_setting(train_file, dom)
            else:
                selected_data = train_file

            for _, line in selected_data.iterrows():
                image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'],
                         line['start_timestamp'], line['stop_timestamp']]
                labels = line['verb_class']
                samples.append((image[0], image[1], image[2], image[3], image[4], int(labels), dom))

        self.split = split
        self.samples = samples
        self.cfg = cfg
        self.cfg_flow = cfg_flow

        self.ops = [
            'AutoContrast', 'Equalize', 'Posterize', 'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness',
            'Rotate', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY'
        ]

    def _select_data_by_setting(self, train_file, dom):
        """Select labeled/unlabeled data based on the semi-supervised setting"""
        if self.semi_setting == 'number':
            return self._select_by_number(train_file, dom)
        elif self.semi_setting == 'ratio':
            return self._select_by_ratio(train_file, dom)
        elif self.semi_setting == 'domain':
            return self._select_by_domain(train_file, dom)
        else:
            raise ValueError(f"Unknown semi_setting: {self.semi_setting}")

    def _select_by_number(self, train_file, dom):
        """Select data based on fixed number per class"""
        grouped = train_file.groupby('verb_class')

        if self.is_labeled:
            labeled_samples = []
            for verb_class, group in grouped:
                if len(group) >= self.semi_value:
                    sampled = group.sample(n=self.semi_value, random_state=42)
                else:
                    sampled = group
                labeled_samples.append(sampled)
                print(f"Domain {dom}, Class {verb_class}: {len(sampled)} labeled samples selected")

            if labeled_samples:
                selected_data = pd.concat(labeled_samples, ignore_index=True)
                print(f"Domain {dom}: Total labeled samples: {len(selected_data)}")
            else:
                selected_data = pd.DataFrame()
        else:
            labeled_indices = set()
            for verb_class, group in grouped:
                if len(group) >= self.semi_value:
                    sampled = group.sample(n=self.semi_value, random_state=42)
                else:
                    sampled = group
                labeled_indices.update(sampled.index)

            selected_data = train_file.drop(labeled_indices)
            print(f"Domain {dom}: Total unlabeled samples: {len(selected_data)}")

        return selected_data

    def _select_by_ratio(self, train_file, dom):
        """Select data based on percentage"""
        total_samples = len(train_file)
        num_labeled_samples = int(total_samples * self.semi_value)

        print(f"Domain {dom}: Total samples: {total_samples}, "
              f"Labeled samples: {num_labeled_samples} ({self.semi_value * 100:.1f}%), "
              f"Unlabeled samples: {total_samples - num_labeled_samples}")

        grouped = train_file.groupby('verb_class')

        if self.is_labeled:
            labeled_samples = []
            for verb_class, group in grouped:
                class_labeled_count = max(1, int(len(group) * self.semi_value))
                class_labeled_count = min(class_labeled_count, len(group))

                if len(group) >= class_labeled_count:
                    sampled = group.sample(n=class_labeled_count, random_state=42)
                else:
                    sampled = group
                labeled_samples.append(sampled)

                print(f"  Class {verb_class}: {len(group)} total -> {len(sampled)} labeled")

            if labeled_samples:
                selected_data = pd.concat(labeled_samples, ignore_index=True)
                print(f"  Final labeled dataset size: {len(selected_data)}")
            else:
                selected_data = pd.DataFrame()
        else:
            labeled_indices = set()
            for verb_class, group in grouped:
                class_labeled_count = max(1, int(len(group) * self.semi_value))
                class_labeled_count = min(class_labeled_count, len(group))

                if len(group) >= class_labeled_count:
                    sampled = group.sample(n=class_labeled_count, random_state=42)
                else:
                    sampled = group
                labeled_indices.update(sampled.index)

            selected_data = train_file.drop(labeled_indices)
            print(f"  Final unlabeled dataset size: {len(selected_data)}")

        return selected_data

    def _select_by_domain(self, train_file, dom):
        """Select data based on domain labeling"""
        if dom in self.unlabeled_domains:
            if self.is_labeled:
                print(f"Domain {dom}: All {len(train_file)} samples are unlabeled, returning empty labeled set")
                return pd.DataFrame()
            else:
                print(f"Domain {dom}: All {len(train_file)} samples are unlabeled")
                return train_file
        else:
            if self.is_labeled:
                print(f"Domain {dom}: All {len(train_file)} samples are labeled")
                return train_file
            else:
                print(f"Domain {dom}: All {len(train_file)} samples are labeled, returning empty unlabeled set")
                return pd.DataFrame()

    def apply_weak_video_augmentation(self, imgs):
        """Apply weak video augmentation."""
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
        """Apply strong video augmentation."""
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
        video_path = self.base_path + 'MM-SADA_Domain_Adaptation_Splits/rgb/' + self.split + '/' + self.samples[index][
            0]
        if self.use_video:
            filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality = self.cfg.data.train.get('modality', 'RGB')
            start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
            data = dict(
                frame_dir=video_path,
                total_frames=int(self.samples[index][2] - self.samples[index][1]),
                label=-1,
                start_index=start_index,
                filename_tmpl=filename_tmpl,
                modality=modality)
            data = self.pipeline(data)

            if self.split == 'train':
                imgs_weak = self.apply_weak_video_augmentation(data['imgs'].clone())

                if not self.is_labeled:
                    imgs_strong = self.apply_strong_video_augmentation(data['imgs'].clone())
                    data['imgs_strong'] = imgs_strong

                data['imgs'] = imgs_weak

            label1 = self.samples[index][-2]

        if self.use_flow:
            flow_path = self.base_path + 'MM-SADA_Domain_Adaptation_Splits/flow/' + self.split + '/' + self.samples[index][
                0]
            # Keep consistent with full-supervised GMP loader.
            filename_tmpl_flow = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
            modality_flow = self.cfg_flow.data.train.get('modality', 'Flow')
            start_index_flow = self.cfg_flow.data.train.get('start_index', int(np.ceil(self.samples[index][1] / 2)))
            flow = dict(
                frame_dir=flow_path,
                total_frames=int((self.samples[index][2] - self.samples[index][1]) / 2),
                label=-1,
                start_index=start_index_flow,
                filename_tmpl=filename_tmpl_flow,
                modality=modality_flow)
            flow = self.pipeline_flow(flow)

            if self.split == 'train':
                flow_weak = self.apply_weak_flow_augmentation(flow['imgs'].clone())
                if not self.is_labeled:
                    flow_strong = self.apply_strong_flow_augmentation(flow['imgs'].clone())
                    flow['imgs_strong'] = flow_strong
                flow['imgs'] = flow_weak

        domain_label = self._get_domain_label(index)

        if self.use_audio:
            audio_path = self.base_path + 'MM-SADA_Domain_Adaptation_Splits/rgb/' + self.split + '/' + \
                         self.samples[index][0] + '.wav'
            samples, samplerate = sf.read(audio_path)

            duration = len(samples) / samplerate

            fr_sec = self.samples[index][3].split(':')
            hour1, minu1, sec1 = float(fr_sec[0]), float(fr_sec[1]), float(fr_sec[2])
            fr_sec = (hour1 * 60 + minu1) * 60 + sec1

            stop_sec = self.samples[index][4].split(':')
            hour1, minu1, sec1 = float(stop_sec[0]), float(stop_sec[1]), float(stop_sec[2])
            stop_sec = (hour1 * 60 + minu1) * 60 + sec1

            start1 = int(np.round(fr_sec / duration * len(samples)))
            end1 = int(np.round(stop_sec / duration * len(samples)))
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
                                label1, domain_label)
                    if self.use_video and self.use_audio:
                        return (data, spectrogram_weak.astype(np.float32),
                                spectrogram_strong.astype(np.float32),
                                label1, domain_label)

            if self.use_video and self.use_flow and self.use_audio:
                return data, flow, spectrogram_weak.astype(np.float32), label1, domain_label
            if self.use_video and self.use_audio:
                return data, spectrogram_weak.astype(np.float32), label1, domain_label
        else:
            print("INPUT ERROR: Both video and audio must be enabled.")
            return None, None, None, None

    def _get_domain_label(self, index):
        """Return the integer domain label for the current source-domain pair."""
        current_domain = self.samples[index][-1]
        if len(self.domain) != 2:
            return torch.tensor(-1)
        domain_mapping = {domain: domain_id for domain_id, domain in enumerate(self.domain)}
        mapped_value = domain_mapping.get(current_domain, -1)
        return torch.tensor(mapped_value)

    def __len__(self):
        return len(self.samples)


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

    labeled_dataset = UnifiedEPICDomainSemiDataset(
        split='train', domain=domains, cfg=cfg, cfg_flow=cfg_flow,
        datapath=datapath, use_video=use_video, use_flow=use_flow, use_audio=use_audio,
        is_labeled=True, semi_setting=semi_setting, semi_value=semi_value,
        unlabeled_domains=unlabeled_domains
    )

    unlabeled_dataset = UnifiedEPICDomainSemiDataset(
        split='train', domain=domains, cfg=cfg, cfg_flow=cfg_flow,
        datapath=datapath, use_video=use_video, use_flow=use_flow, use_audio=use_audio,
        is_labeled=False, semi_setting=semi_setting, semi_value=semi_value,
        unlabeled_domains=unlabeled_domains
    )

    return labeled_dataset, unlabeled_dataset
