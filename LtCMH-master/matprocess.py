import scipy.io
import pandas as pd
import numpy as np
import os
import h5py
import time


class RenameMat:
    def __init__(self):
        self.path = './origin/'
        self.target = './sample/'
        self.mat_list = []
        self.mat_data = None
        self.sample_result = []
        self.sample_dit = {
            '24': {
                #sample strategy
            },
            '21': {
                # sample strategy
            }
        }

    def get_mk_target_dir(self):
        if not os.path.exists(self.target):
            os.mkdir(self.target)

    def get_read_mat_files(self):
        for file in os.listdir(self.path):
            self.mat_list.append(file)
        return self.mat_list

    def get_load_mat_file(self, mat_file_path):
        try:
            self.mat_data = scipy.io.loadmat(mat_file_path)
        except:
            self.mat_data = h5py.File(mat_file_path)
        return self.mat_data

    @staticmethod
    def bisect_right(a, x, lo=0, hi=None):
        """Return the index where to insert item x in list a, assuming a is sorted.

        The return value i is such that all e in a[:i] have e <= x, and all e in
        a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
        insert just after the rightmost x already there.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def get_l_tr_samples(self):
        l_tr = self.mat_data['L_tr'][:]
        if l_tr.shape[0] < l_tr.shape[1]:
            dfx = pd.DataFrame(l_tr).T
        else:
            dfx = pd.DataFrame(l_tr)
        # print(dfx.columns)
        dfx.loc['col_sum'] = dfx.apply(lambda x: x.sum(), axis=0)
        # print('L_tr', dfx)
        df_sum = pd.Series(dfx.loc['col_sum', :])
        df_sum.sort_values(inplace=True, ascending=False)


        self.sample_result = []
        for k, [index, _] in enumerate(df_sum.iteritems()):
            # print(k, index, value)

            type_k = str(len(dfx.columns))
            list_ = self.sample_dit[type_k]['0']
            type_sub_k = self.bisect_right(list_, k)
            sample_num = self.sample_dit[type_k][type_sub_k]

            # print(index, type_k, type_sub_k, sample_num)

            df_origin_sample = pd.DataFrame(dfx[index][:-1])
            df_origin_sample.reset_index(inplace=True)
            df_origin_sample.columns = ['index', 'use_col']
            df_origin_sample.query('use_col=="1"', inplace=True)
            # print(df_origin_sample)
            sample_list = list(pd.DataFrame(df_origin_sample.sample(n=sample_num, replace=False))['index'])
            # print(sample_list)
            # print(len(sample_list))

            self.sample_result = self.sample_result + sample_list

        self.sample_result = list(set(self.sample_result))
        self.sample_result.sort()
        print('sample result：【', len(self.sample_result), '】')
        print('sample result：', self.sample_result)
        return self.sample_result


def run():
    t = time.time()
    s = RenameMat()
    s.get_mk_target_dir()
    s.get_read_mat_files()
    for mat in s.mat_list:
        print('//' * 100)
        file_path = s.path + mat
        s.get_load_mat_file(file_path)
        # l tr
        s.get_l_tr_samples()
        ''' <KeysViewHDF5 ['I_te', 'I_tr', 'L_te', 'L_tr', 'T_te', 'T_tr']>  '''
        result_mat_dit = {
            'I_te': None,
            'I_tr': None,
            'L_te': None,
            'L_tr': None,
            'T_te': None,
            'T_tr': None,
        }
        for k, v in result_mat_dit.items():
            print('+' * 100)
            print('key:', k)
            value_list = s.mat_data[k][:]
            if len(value_list) < len(value_list[0]):
                value_list = value_list.T

            if 'tr' in k:
                df_origin = pd.DataFrame(value_list)
                df_origin.reset_index(inplace=True)
                df_s = pd.DataFrame(s.sample_result)
                df_s.columns = ['index']

                df_sample = pd.merge(left=df_s, right=df_origin, how='left', on='index')
                df_sample.drop(columns='index', inplace=True)

                print(df_sample)
            else:
                df_sample = value_list
                print(df_sample)

            if 'I_' in k:
                df_sample = np.array(df_sample, dtype='single')
            else:
                df_sample = np.array(df_sample, dtype='double')

            result_mat_dit[k] = np.array(df_sample)
        scipy.io.savemat(s.target + mat, result_mat_dit)
        # break
    print(time.time() - t)


if __name__ == '__main__':
    run()
