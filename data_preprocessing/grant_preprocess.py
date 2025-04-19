import sys
import os
import glob
import csv
import pandas as pd
import numpy as np
import argparse

## 시나리오 폴더 단위 읽기, 전처리를 함수 형태로 작성
# 한 객체 단위 -> 한 시나리오 단위로 확장(시나리오의 각 객체별로 csv 파일 저장)
def data_preprocess_scenario(path, scenario):
    print("processing scenario : ",scenario)
    path = path+scenario+'/'# path - 한 시나리오 폴더(19 객체 전부)
    outpath = './grant/VT-MAK_data/data_preprocessed/' # 전처리 된 csv가 저장될 위치 (객체별 저장)
    # aug_outpath = './VT-MAK_data/data_augmented/'
    folders = os.listdir(path)
    out_folders = os.listdir(outpath)
    os.makedirs(outpath+scenario, exist_ok=True)
    # print(folders)


    for folder in folders: # 한 시나리오의 19 객체 loop
        one_entity_csv = pd.DataFrame(columns=['Time', 'EntityID', 'EntityType', 'Force', 'Location_y', 'Location_z', 'Health']) # 객체별 csv 통합 파일 저장을 위해 정의
        augmented_data = pd.DataFrame()
        folderName = folder # 각 폴더명 저장

        files = os.listdir(path + folder) # 불러온 파일들을 리스트에 저장
        files = [file_csv for file_csv in files if file_csv.endswith(".csv")] # csv 파일만 불러옴

        for file in files: # csv 파일 단위(90초) loop
            fileName = file # 각 파일명 저장
            all_csv_files = path + folderName + '/' + fileName # 폴더와 파일 이름 조합
            ######  #EntityStatePDU 누락 데이터 있을 시 아래 소스 주석 해제해서 파일 확인

            #print(folderName)
            #print(file)
            
            
            # 에러 처리
            f = open(all_csv_files)
            reader = csv.reader(f)
            csv_list = []
            for l in reader:
                csv_list.append(l)
            f.close() 
            
            # preprocess
            data_raw = pd.DataFrame(csv_list)
            data_raw = data_raw.loc[:31, :] # detection information 제거
            data_raw = data_raw.transpose() # transpose

            data_raw.rename(columns=data_raw.iloc[0], inplace=True) # col 이름 설정
            data_raw = data_raw.drop(data_raw.index[0])
            #print(folder)
            entity_type = data_raw['EntityType'].iloc[0]
            #print(entity_type)
            data_raw['EntityType'] = str(entity_type).split(':')[0] #+":" + str(entity_type).split(':')[1]
            if data_raw['EntityType'].iloc[0] == "3" :
                data_raw = data_raw.loc[:, ['TimeStamp(sec)', 'EntityID', 'EntityType', 'Force', 'Location(GeoCentric X-Y-Z in (m))', 'Health']]
            elif data_raw['EntityType'].iloc[0] == "1" :
                data_raw = data_raw.loc[:, ['TimeStamp(sec)', 'EntityID', 'EntityType', 'Force', 'Location(GeoCentric X-Y-Z in (m))', 'Damage']]
                data_raw = data_raw.rename(columns={'Damage' : 'Health'})
            data_raw = data_raw.reset_index(drop=True) # 인덱스 0부터 되게 초기화
            data_raw = data_raw.rename(columns={'TimeStamp(sec)' : 'Time'}) # Time col 이름 변경


            df_preprocessed = data_raw.drop(['Location(GeoCentric X-Y-Z in (m))'], axis=1) # location : (x,y,z)
            location = data_raw['Location(GeoCentric X-Y-Z in (m))']
            location = location.drop(index=0) # x좌표 제거
            location = location.dropna()
            location = location.reset_index(drop=True)

            df_preprocessed['Location_y'] = location # 위치 col 이름 설정
            df_preprocessed['Location_z'] = location
            loc_y = location.loc[0]
            loc_z = location.loc[1]
            df_preprocessed['Location_y'].loc[0] = loc_y 
            df_preprocessed['Location_z'].loc[0] = loc_z
            df_preprocessed = df_preprocessed.loc[:0, :] # 필요없는 행 제거
            df_preprocessed = df_preprocessed[['Time', 'EntityID', 'EntityType', 'Force', 'Location_y', 'Location_z', 'Health']] # col 순서 변경
            df_preprocessed['Time'] = float(df_preprocessed['Time']) # time col 값 실수 처리

            one_entity_csv = pd.concat([one_entity_csv, df_preprocessed])
            
            print(df_preprocessed)

    
            # # Augmentation ## 일부 시나리오만 제대로 적용됨
            # augmented_data = one_entity_csv
            # force = augmented_data['Force']
            # if str(force.iloc[0]) == 'Friendly':
            #     force.replace('Friendly', 'Opposing', inplace=True)
            # elif str(force.iloc[0]) == 'Opposing':
            #     force.replace('Opposing', 'Friendly', inplace=True)
     

        
        one_entity_csv = one_entity_csv.sort_values(by=['Time']) # time 순으로 정렬
        one_entity_csv = one_entity_csv.reset_index(drop=True) # index 재설정
        # one_entity_csv.to_csv(outpath+scenario+'/'+folder+".csv",index=False,encoding="utf-8-sig")

        # augmented_data = augmented_data.sort_values(by=['Time']) # time 순으로 정렬
        # augmented_data = augmented_data.reset_index(drop=True) # index 재설정
        # augmented_data.to_csv(aug_outpath+scenario+'/'+folder+".csv",index=False,encoding="utf-8-sig") # augmented data path


def preprocessing_main(args):
    path = args.path
    folders = os.listdir(path)
    for scenario in folders:
        data_preprocess_scenario(path,scenario)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Recommend Model')
    parser.add_argument('--path', type=str, default='./grant/VT-MAK_data/data_original/')


    args = parser.parse_args()
    preprocessing_main(args)