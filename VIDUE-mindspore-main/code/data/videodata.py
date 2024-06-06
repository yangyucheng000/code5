import os
import glob
import numpy as np
import imageio
import utils.utils as utils
import cv2
import random

class VIDEODATA:
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.in_seq = args.n_sequence
        self.n_seq = args.n_sequence * args.n_outputs
        self.n_frames_per_video = args.n_frames_per_video
        self.xN = args.n_outputs   #
        self.random = args.random
        self.batch = args.batch_size
        if self.random:
            self.exposure = [i+1 for i in range(self.xN//2, self.xN, 2)]
            self.readout = [self.xN-j for j in self.exposure]
            self.curr_exposure = self.exposure[0]
            self.p = args.p
        else:
            self.exposure = args.m
            self.readout = args.n
        print("n_seq:", self.n_seq)
        print("exposure:", self.exposure, "readout:", self.readout)
        print("n_frames_per_video:", args.n_frames_per_video)

        self.n_frames_video = []
        if train:
            self.apath = args.dir_data
        else:
            self.apath = args.dir_data_test

        self.images = self._scan()
        # random.shuffle(self.images)

        self.num_video = len(self.images)
        self.num_frame = sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)
        print("Number of videos to load:", self.num_video)
        print("Number of frames to load:", self.num_frame)

        if train:
            self.repeat = max(args.test_every // max((self.num_frame // self.args.batch_size), 1), 1)
            print("Dataset repeat:", self.repeat)

        if args.process:
            self.data_images = self._load(self.images)

    def _set_filesystem(self, dir_data):
        print("Loading {} => {} DataSet".format("train" if self.train else "test", self.name))
        self.apath = dir_data

    def _scan(self):
        vid_names = sorted(glob.glob(os.path.join(self.apath, '*')))   #[0:1]

        images = []

        for i in range(len(vid_names)):
            if self.train:
                dir_names = sorted(glob.glob(os.path.join(vid_names[i], '*')))[:self.args.n_frames_per_video]
            else:
                dir_names = sorted(glob.glob(os.path.join(vid_names[i], '*')))
            images.append(dir_names)
            self.n_frames_video.append(len(dir_names))

        return images

    def _load(self, images):
        data_images = []

        n_videos = len(images)
        for idx in range(n_videos):
            if idx % 10 == 0:
                print("Loading video %d" % idx)
            gts = [imageio.imread(hr_name) for hr_name in images[idx]]
            gts = np.array([cv2.resize(frame, None, fx=0.5, fy=0.5) for frame in gts])
            print(gts.shape)
            data_images.append(gts)

        return data_images

    def __getitem__(self, idx):

        if self.train:
            if self.args.process:
                inputs, gts, output_filenames, input_filenames, exposure = self._load_file_from_loaded_data(idx)
            else:
                inputs, gts, output_filenames, input_filenames, exposure = self._load_file(idx)
        else:
            if self.args.process:
                inputs, gts, output_filenames, input_filenames, exposure = self._load_file_from_loaded_data_test(idx)
            else:
                inputs, gts, output_filenames, input_filenames, exposure = self._load_file_test(idx)

        inputs_list = [inputs[i, :, :, :] for i in range(self.n_seq // self.xN)]
        inputs_concat = np.concatenate(inputs_list, axis=2)
        gts_list = [gts[i, :, :, :] for i in range(self.xN)]
        gts_concat = np.concatenate(gts_list, axis=2)

        inputs_concat, gts_concat = self.get_patch(inputs_concat, gts_concat, self.args.size_must_mode)  #, bms_concat     , bms_concat
        inputs_list = [inputs_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.n_seq // self.xN)]
        gts_list = [gts_concat[:, :, i*self.args.n_colors:(i+1)*self.args.n_colors] for i in range(self.xN)]

        inputs = np.array(inputs_list)
        gts = np.array(gts_list)

        input_tensors = utils.np2Tensor(*inputs, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        gt_tensors = utils.np2Tensor(*gts, rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
        
        return input_tensors, gt_tensors, output_filenames, input_filenames, exposure

    def __len__(self):
        if self.train:
            return self.num_frame #* self.repeat
        else:
            return self.num_frame // self.xN + 1

    def _get_index(self, idx):
        if self.train:
            return idx % self.num_frame
        else:
            return idx * self.xN

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        if self.random:
            self.curr_exposure = np.random.choice(self.exposure, p=np.array(self.p))
        else:
            self.curr_exposure = self.exposure

        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        # random.shuffle(n_poss_frames)
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        if self.in_seq % 2 == 0:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq // 2-1):frame_idx + self.xN * (self.in_seq // 2)]
        else:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq//2):frame_idx + self.xN * (self.in_seq//2 +1)]
        f_inputs = self.images[video_idx][frame_idx+self.curr_exposure//2:frame_idx + self.n_seq:self.xN]
        inputs = []
        frame_gt = imageio.imread(f_gts[0])
        frame_gt = cv2.resize(frame_gt, None, fx=0.5, fy=0.5)
        gt_img = np.array(frame_gt)
        H, W, C = gt_img.shape
        for i in range(frame_idx+self.curr_exposure//2,frame_idx + self.n_seq,self.xN):
            blur = np.zeros((H, W, C))
            for j in range(i-self.curr_exposure//2,i+self.curr_exposure//2+1):
                frame_j = imageio.imread(self.images[video_idx][j])
                frame_j = cv2.resize(frame_j, None, fx=0.5, fy=0.5)
                blur += np.array(frame_j) / self.curr_exposure
            inputs.append(blur)

        gts = [imageio.imread(hr_name) for hr_name in f_gts]
        gts = np.array([cv2.resize(gt, None, fx=0.5, fy=0.5) for gt in gts])
        inputs = np.array(inputs)
        output_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]
        input_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                            for name in f_inputs]

        return inputs, gts, output_filenames, input_filenames, self.curr_exposure  #, bms, f_labels

    def _load_file_from_loaded_data(self, idx):
        if self.random:
            # if idx % self.batch == 0:
            #     self.curr_exposure = random.choice(self.exposure)
            # idx = np.random.randint(0, self.num_frame)
            # self.curr_exposure = random.choice(self.exposure)
            self.curr_exposure = np.random.choice(self.exposure, p=np.array(self.p))
        else:
            self.curr_exposure = self.exposure

        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        # random.shuffle(n_poss_frames)
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        if self.in_seq % 2 == 0:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq // 2-1):frame_idx + self.xN * (self.in_seq // 2)]
        else:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq//2):frame_idx + self.xN * (self.in_seq//2 +1)]
        f_inputs = self.images[video_idx][frame_idx+self.curr_exposure//2:frame_idx + self.n_seq:self.xN]
        inputs = []
        H, W, C = self.data_images[0][0].shape
        # print("exp:",self.curr_exposure)
        for i in range(frame_idx+self.curr_exposure//2,frame_idx + self.n_seq,self.xN):
            blur = np.zeros((H, W, C))
            for j in range(i-self.curr_exposure//2,i+self.curr_exposure//2+1):
                frame_j = self.data_images[video_idx][j]
                blur += np.array(frame_j) / self.curr_exposure
            inputs.append(blur)
            # print("blur:", i-self.curr_exposure//2,i+self.curr_exposure//2)

        if self.in_seq % 2 == 0:
            gts = self.data_images[video_idx][
                    frame_idx + self.xN * (self.in_seq // 2 - 1):frame_idx + self.xN * (self.in_seq // 2)]
            # print("gt:", frame_idx + self.xN * (self.in_seq // 2 - 1),frame_idx + self.xN * (self.in_seq // 2))
        else:
            gts = self.data_images[video_idx][
                    frame_idx + self.xN * (self.in_seq // 2):frame_idx + self.xN * (self.in_seq // 2 + 1)]
            # print("gt:", frame_idx + self.xN * (self.in_seq // 2), frame_idx + self.xN * (self.in_seq // 2 + 1))
        gts = np.array(gts)
        inputs = np.array(inputs)
        output_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]
        input_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                            for name in f_inputs]

        return inputs, gts, output_filenames, input_filenames, self.curr_exposure  #, bms, f_labels

    def _load_file_test(self, idx):
        if self.random:
            kinds = len(self.exposure)
            exposure = self.exposure[idx%kinds]
        else:
            exposure = self.exposure

        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        if self.in_seq % 2 == 0:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq // 2-1):frame_idx + self.xN * (self.in_seq // 2)]
        else:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq//2):frame_idx + self.xN * (self.in_seq//2 +1)]
        f_inputs = self.images[video_idx][frame_idx+exposure//2:frame_idx + self.n_seq:self.xN]
        inputs = []
        frame_gt = imageio.imread(f_gts[0])
        frame_gt = cv2.resize(frame_gt, None, fx=0.5, fy=0.5)
        gt_img = np.array(frame_gt)
        H, W, C = gt_img.shape
        for i in range(frame_idx+exposure//2,frame_idx + self.n_seq,self.xN):
            blur = np.zeros((H, W, C))
            for j in range(i-exposure//2,i+exposure//2+1):
                frame_j = imageio.imread(self.images[video_idx][j])
                frame_j = cv2.resize(frame_j, None, fx=0.5, fy=0.5)
                blur += np.array(frame_j) / exposure
                # print("Generate Blur:", self.images[video_idx][j])
            inputs.append(blur)

        gts = [imageio.imread(hr_name) for hr_name in f_gts]
        gts = np.array([cv2.resize(gt, None, fx=0.5, fy=0.5) for gt in gts])
        inputs = np.array(inputs)
        output_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]
        input_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                            for name in f_inputs]

        return inputs, gts, output_filenames, input_filenames, exposure

    def _load_file_from_loaded_data_test(self, idx):
        if self.random:
            kinds = len(self.exposure)
            exposure = self.exposure[idx%kinds]
        else:
            exposure = self.exposure

        idx = self._get_index(idx)

        n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
        video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
        if self.in_seq % 2 == 0:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq // 2-1):frame_idx + self.xN * (self.in_seq // 2)]
        else:
            f_gts = self.images[video_idx][frame_idx + self.xN * (self.in_seq//2):frame_idx + self.xN * (self.in_seq//2 +1)]
        f_inputs = self.images[video_idx][frame_idx+exposure//2:frame_idx + self.n_seq:self.xN]
        inputs = []
        H, W, C = self.data_images[0][0].shape
        for i in range(frame_idx+exposure//2,frame_idx + self.n_seq,self.xN):
            blur = np.zeros((H, W, C))
            for j in range(i-exposure//2,i+exposure//2+1):
                frame_j = self.data_images[video_idx][j]
                blur += np.array(frame_j) / exposure
                # print("Generate Blur:", self.images[video_idx][j])
            inputs.append(blur)

        if self.in_seq % 2 == 0:
            gts = self.data_images[video_idx][
                    frame_idx + self.xN * (self.in_seq // 2 - 1):frame_idx + self.xN * (self.in_seq // 2)]
        else:
            gts = self.data_images[video_idx][
                    frame_idx + self.xN * (self.in_seq // 2):frame_idx + self.xN * (self.in_seq // 2 + 1)]
        gts = np.array(gts)
        inputs = np.array(inputs)
        output_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                     for name in f_gts]
        input_filenames = [os.path.split(os.path.dirname(name))[-1] + '.' + os.path.splitext(os.path.basename(name))[0]
                            for name in f_inputs]

        return inputs, gts, output_filenames, input_filenames, exposure

    def get_patch(self, input, gt, size_must_mode=1):   #, bm
        if self.train:
            input, gt = utils.get_patch(input, gt, patch_size=self.args.patch_size)   #, bm    , bm
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]   #, bm     , bm[:new_h, :new_w, :]
            if not self.args.no_augment:
                input, gt = utils.data_augment(input, gt)   #, bm     , bm
        else:
            h, w, c = input.shape
            new_h, new_w = h - h % size_must_mode, w - w % size_must_mode
            input, gt = input[:new_h, :new_w, :], gt[:new_h, :new_w, :]   #, bm     , bm[:new_h, :new_w, :]
        return input, gt
