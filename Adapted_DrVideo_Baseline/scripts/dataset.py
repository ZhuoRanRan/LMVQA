# dataset.py — Open-ended dataset loader
# For generic 'egoschema-like' JSON inputs (kept for compatibility).
# Lecture-mode CSV loading is handled in main.py (per-CSV iteration).

from util import save_json, load_json, save_pkl, load_pkl, makedir, parse_args
from torch.utils.data import Dataset
from os.path import join, expanduser


class BaseDataset(Dataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.args = args
        self.narrations = self.get_descriptions()  # uid --> list[str] or str
        self.anno = self.get_anno()                # uid --> {"question": str}
        self.durations = load_json(args.duration_path)  # uid --> seconds
        data = self.build()
        data = self.filter(data, quids_to_exclude, num_examples_to_run)
        self.data = data

    def set_ukey(self, name):
        self.ukey = name

    def filter(self, data, quids_to_exclude, num_examples_to_run):
        if quids_to_exclude is not None:
            data = [el for el in data if el[self.ukey] not in quids_to_exclude]
        if num_examples_to_run >= 0:
            data = data[:num_examples_to_run]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EgoSchemaDataset(BaseDataset):
    def __init__(self, args, quids_to_exclude=None, num_examples_to_run=-1):
        self.set_ukey('uid')
        super().__init__(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

    def get_descriptions(self):
        narrations = load_json(self.args.data_path)  # uid -> list[str] or str
        return narrations
    
    def format_narration(self, narr):
        if isinstance(narr, list):
            narr = '. '.join(narr)
        return narr

    def get_anno(self):
        # Only require 'question' per uid for open-ended
        anno = load_json(self.args.anno_path)  # uid --> { "question": "..." , ... }
        return anno

    def build(self):
        data = []
        for uid, item in self.anno.items():
            if uid not in self.narrations:
                continue
            question = item['question']
            duration = int(self.durations.get(uid, 0))
            video_root = expanduser(getattr(self.args, 'video_root', '~/Videos/drvideo'))
            video_path = join(video_root, uid + ".mp4")
            data.append({
                'uid': uid,
                'video_path': video_path,
                'question': question,
                'duration': duration,
            })
        return data


def get_dataset(args, quids_to_exclude=None, num_examples_to_run=-1):
    return EgoSchemaDataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=num_examples_to_run)

if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args, num_examples_to_run=args.num_examples_to_run)
    print(len(dataset))
