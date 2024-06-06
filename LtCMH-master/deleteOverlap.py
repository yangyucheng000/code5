import pandas as pd
import os
import shutil
from tqdm import tqdm
import re

class DataDelete:
    def __init__(self):
        self.origin_path='F:\experiment\data\\'
        self.target_path='F:\experiment\data\\'

    def get_copy_origin_files(self):
        if os.path.exists(self.target_path):
            shutil.rmtree(self.target_path)
        print('copying...')
        for sample_file in tqdm(os.listdir(self.origin_path)):
            origin_path = self.origin_path+sample_file
            target_path=self.origin_path+sample_file
            shutil.copytree(origin_path,target_path)

    def get_tags_size(self,sample_path):
        path_tags=self.target_path + f'{sample_path}/tags/'
        path_img=self.target_path+f'{sample_path}/img/'
        xlsx_name=[index for index in os.listdir(self.target_path+f'{sample_path}/')][0]
        path_xlsx=self.target_path+f'{sample_path}/'+xlsx_name

        tags_list=[]
        delete_tags_list = []
        for file_name in os.listdir(path_tags):
            # size=os.path.getsize(path_tags+file_name)
            tag_row_count=len(open(path_tags+file_name,'r',encoding='utf-8').readlines())
            name_num=re.findall('tags([0-9]+).txt',file_name)[0]
            tags_list.append(eval(name_num))

            if tag_row_count<=20 :
                delete_tags_list.append(eval(name_num))
                img_name='im' +name_num + '.jpg'
                os.remove(path_tags+file_name)
                os.remove(path_img+img_name)

        tags_list.sort()
        delete_tags_list.sort()
        print(tags_list)
        print(delete_tags_list)

        delete_index_list=[]
        for delete_tag in delete_index_list:
            delete_tags_list.append(tags_list.index(delete_tag))
        delete_index_list.sort()
        print(delete_index_list)

        df_sample =pd.read_excel(path_xlsx,header=None)
        df_sample.reset_index(inplace=True)
        print(df_sample)

        df_sample['x']= df_sample.apply(lambda x:'yes' if x['index'] in delete_index_list else 'no',axis=1 )
        df_result= df_sample.query('x=="no"')
        df_result.reset_index(drop=True,inplace=True)
        df = df_result.drop(['index','x'],axis=1)
        df.to_excel(path_xlsx,header=None,index=None,encoding='gb18030')
        print(df)

def run():
    parse = DataDelete()
    # parse.get_copy_origin_files()

    for sample_path in os.listdir(parse.target_path):
        print('-'*50)
        print(sample_path)
        parse.get_tags_size(sample_path)


if __name__ == '__main__':
    run()


