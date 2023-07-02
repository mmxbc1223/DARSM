import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler
from global_configs import *
class MultimodalDataset(Dataset):
    def __init__(self, data, tokenizer=None):
        self.sentences = []
        self.input_ids = []
        self.attention_masks = []
        self.visuals = []
        self.acoustics = []
        self.labels = []
        self.segments = []
        self.visual_length = []
        self.acoustic_length = []
        for content in data:
            (sentence, visual, acoustic), label_id, segment = content
            self.sentences.append(sentence)
            inputs = tokenizer([sentence], return_tensors="pt", max_length=50, truncation=True, padding='max_length')
            self.input_ids.append(inputs['input_ids'].squeeze(0))
            self.attention_masks.append(inputs['attention_mask'].squeeze(0))
            visual = torch.tensor(visual.astype(np.float32)).cpu().detach()
            self.visuals.append(visual)
            visual_len = visual.shape[0]
            self.visual_length.append(visual_len)

            acoustic = torch.tensor(acoustic.astype(np.float32)).cpu().detach()
            self.acoustics.append(acoustic)
            acoustic_len = acoustic.shape[0]
            self.acoustic_length.append(acoustic_len)


            label_id = torch.tensor(label_id.astype(np.float32)).cpu().detach()
            self.labels.append(label_id)
            self.segments.append(segment)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        input_ids = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        visual = self.visuals[idx]
        label_id = self.labels[idx]
        segment = self.segments[idx]
        visual_len = self.visual_length[idx]
        acoustic = self.acoustics[idx]
        acoustic_len = self.acoustic_length[idx]
        return sentence, input_ids, attention_mask, visual, visual_len, acoustic, acoustic_len, label_id, segment