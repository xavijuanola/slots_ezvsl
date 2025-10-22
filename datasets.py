import os
import csv
import librosa
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from scipy import signal
import random
import json
import xml.etree.ElementTree as ET
from audio_io import load_audio_av, open_audio_av


def load_image(path):
    return Image.open(path).convert('RGB')


def load_spectrogram(path, dur=3.):
    # Load audio
    # audio_ctr = open_audio_av(path)
    # audio_dur = audio_ctr.streams.audio[0].duration * audio_ctr.streams.audio[0].time_base
    # audio_ss = max(float(audio_dur)/2 - dur/2, 0)
    # audio, samplerate = load_audio_av(container=audio_ctr, start_time=audio_ss, duration=dur)
    audio, samplerate = librosa.load(path, sr=16000)

    # To Mono
    audio = np.clip(audio, -1., 1.)

    # Repeat if audio is too short
    if audio.shape[0] < samplerate * dur:
        n = int(samplerate * dur / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio = audio[:int(samplerate * dur)]

    frequencies, times, spectrogram = signal.spectrogram(audio, samplerate, nperseg=512, noverlap=274)
    spectrogram = np.log(spectrogram + 1e-7)
    return spectrogram

def load_all_bboxes(annotation_dir, format='flickr'):
    gt_bboxes = {}
    if format == 'flickr':
        anno_files = os.listdir(annotation_dir)
        for filename in anno_files:
            file = filename.split('.')[0]
            gt = ET.parse(f"{annotation_dir}/{filename}").getroot()
            bboxes = []
            for child in gt:
                for childs in child:
                    bbox = []
                    if childs.tag == 'bbox':
                        for index, ch in enumerate(childs):
                            if index == 0:
                                continue
                            bbox.append(int(224 * int(ch.text)/256))
                    bboxes.append(bbox)
            gt_bboxes[file] = bboxes

    elif format == 'vggss':
        with open('metadata/vggss.json') as json_file:
            annotations = json.load(json_file)
        for annotation in annotations:
            bboxes = [(np.clip(np.array(bbox), 0, 1) * 224).astype(int) for bbox in annotation['bbox']]
            gt_bboxes[annotation['file']] = bboxes

    return gt_bboxes


def bbox2gtmap(bboxes, format='flickr'):
    gt_map = np.zeros([224, 224])
    for xmin, ymin, xmax, ymax in bboxes:
        temp = np.zeros([224, 224])
        temp[ymin:ymax, xmin:xmax] = 1
        gt_map += temp

    if format == 'flickr':
        # Annotation consensus
        gt_map = gt_map / 2
        gt_map[gt_map > 1] = 1

    elif format == 'vggss':
        # Single annotation
        gt_map[gt_map > 0] = 1

    return gt_map


class AudioVisualDataset(Dataset):
    def __init__(self, image_files, audio_files, image_path, audio_path, audio_dur=3., mode='train', image_transform=None, audio_transform=None, all_bboxes=None, bbox_format='flickr'):
        super().__init__()
        
        self.mode = mode
        self.audio_path = audio_path
        self.image_path = image_path

        self.audio_dur = audio_dur

        self.audio_files = audio_files
        self.image_files = image_files
        self.all_bboxes = all_bboxes
        self.bbox_format = bbox_format

        self.image_transform = image_transform
        self.audio_transform = audio_transform

    def getitem(self, idx):
        file = self.image_files[idx]
        file_id = file.split('.')[0]

        # Image
        if self.mode == 'train':
            img_fn = os.path.join(self.image_path, self.image_files[idx].split('.')[0], '125.jpg')
        else:    
            img_fn = os.path.join(self.image_path, self.image_files[idx])
        frame = self.image_transform(load_image(img_fn))

        # Audio
        audio_fn = os.path.join(self.audio_path, self.audio_files[idx])
        spectrogram = self.audio_transform(load_spectrogram(audio_fn))

        bboxes = {}
        if self.all_bboxes is not None:
            bboxes['bboxes'] = self.all_bboxes[file_id]
            bboxes['gt_map'] = bbox2gtmap(self.all_bboxes[file_id], self.bbox_format)

        return frame, spectrogram, bboxes, file_id

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception:
            return self.getitem(random.sample(range(len(self)), 1)[0])

def get_train_test_dataset(args):
    image_path = f"{args.train_data_path}/total_video_frames/"
    audio_path = f"{args.train_data_path}/total_video_3s_audio/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn for fn in os.listdir(image_path)}
    total_files = audio_files.intersection(image_files)
    print(f"{len(total_files)} total available files")

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        train_data = total_files.intersection(subset)
        print(f"{len(train_data)} valid subset files")
    train_data = sorted(list(train_data))

    # Get 5000 random files from total_files that are not in avail_files
    remaining_files = list(total_files - set(train_data))
    random.shuffle(remaining_files)
    val_data = remaining_files[:5000]

    if args.debug == 'True':
        train_data = train_data[:500]
        val_data = val_data[:50]

    args.train_log_files = ['1ksINuMyHV4_000014', '0kGVS6nRjgA_000091']
    args.val_log_files = ['7bdDCI8Q3mY_000208', 'WTMrgwvE84o_000000']

    # Ensure train_log_files are in train_data and NOT in val_data
    for log_file in args.train_log_files:
        if log_file in val_data:
            val_data.remove(log_file)
        if log_file not in train_data:
            train_data.insert(0, log_file)

    # Ensure val_log_files are in val_data and NOT in train_data
    for log_file in args.val_log_files:
        if log_file in train_data:
            train_data.remove(log_file)
        if log_file not in val_data:
            val_data.insert(0, log_file)
            
    audio_files_train = sorted([dt+'.wav' for dt in train_data])
    image_files_train = sorted([dt+'.jpg' for dt in train_data])

    audio_files_val = sorted([dt+'.wav' for dt in val_data])
    image_files_val = sorted([dt+'.jpg' for dt in val_data])
    
    # Transforms
    image_transform_train = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        # transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        # transforms.RandomCrop((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])
    
    image_transform_val = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])
    
    train_dataset = AudioVisualDataset(
        image_files=image_files_train,
        audio_files=audio_files_train,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        mode='train',
        image_transform=image_transform_train,
        audio_transform=audio_transform_train
    )

    val_dataset = AudioVisualDataset(
        image_files=image_files_val,
        audio_files=audio_files_val,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        mode='train',
        image_transform=image_transform_val,
        audio_transform=audio_transform_val
    )

    return train_dataset, val_dataset

def get_train_dataset(args):
    image_path = f"{args.train_data_path}/total_video_frames/"
    audio_path = f"{args.train_data_path}/total_video_3s_audio/"

    # List directory
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path) if fn.endswith('.wav')}
    image_files = {fn for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    print(f"{len(avail_files)} available files")

    # Subsample if specified
    if args.trainset.lower() in {'vggss', 'flickr'}:
        pass    # use full dataset
    else:
        subset = set(open(f"metadata/{args.trainset}.txt").read().splitlines())
        avail_files = avail_files.intersection(subset)
        print(f"{len(avail_files)} valid subset files")
    avail_files = sorted(list(avail_files))

    if args.debug == 'True':
        avail_files = avail_files[:10]

    audio_files = sorted([dt+'.wav' for dt in avail_files])
    image_files = sorted([dt+'.jpg' for dt in avail_files])

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize(int(224 * 1.1), Image.BICUBIC),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        mode='train',
        image_transform=image_transform,
        audio_transform=audio_transform
    )


def get_test_dataset(args):
    audio_path = args.test_data_path + '/audios_correct/'
    image_path = args.test_data_path + '/frames_correct/'

    if args.testset == 'flickr':
        testcsv = 'metadata/flickr_test.csv'
    elif args.testset == 'vggss':
        testcsv = 'metadata/vggss_test.csv'
    elif args.testset == 'vggss_heard':
        testcsv = 'metadata/vggss_heard_test.csv'
    elif args.testset == 'vggss_unheard':
        testcsv = 'metadata/vggss_unheard_test.csv'
    else:
        raise NotImplementedError
    bbox_format = {'flickr': 'flickr',
                   'vggss': 'vggss',
                   'vggss_heard': 'vggss',
                   'vggss_unheard': 'vggss'}[args.testset]

    #  Retrieve list of audio and video files
    testset = set([item[0] for item in csv.reader(open(testcsv))])

    if args.debug == 'True':
        testset = set(list(testset)[:10])

    # Intersect with available files
    audio_files = {fn.split('.wav')[0] for fn in os.listdir(audio_path)}
    image_files = {fn.split('.jpg')[0] for fn in os.listdir(image_path)}
    avail_files = audio_files.intersection(image_files)
    testset = testset.intersection(avail_files)

    testset = sorted(list(testset))
    image_files = [dt+'.jpg' for dt in testset]
    audio_files = [dt+'.wav' for dt in testset]

    # Bounding boxes
    all_bboxes = load_all_bboxes(args.test_gt_path, format=bbox_format)

    # Transforms
    image_transform = transforms.Compose([
        transforms.Resize((224, 224), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    audio_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.0], std=[12.0])])

    return AudioVisualDataset(
        image_files=image_files,
        audio_files=audio_files,
        image_path=image_path,
        audio_path=audio_path,
        audio_dur=3.,
        mode='val',
        image_transform=image_transform,
        audio_transform=audio_transform,
        all_bboxes=all_bboxes,
        bbox_format=bbox_format
    )


def inverse_normalize(tensor):
    inverse_mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
    inverse_std = [1.0/0.229, 1.0/0.224, 1.0/0.225]
    tensor = transforms.Normalize(inverse_mean, inverse_std)(tensor)
    return tensor



