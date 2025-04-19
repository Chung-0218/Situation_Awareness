import sys
import os
import glob
import csv
import pandas as pd
import numpy as np
import argparse

## csv 파일 1개만 읽기
# 에러 처리
# f = open('VT-MAK_DATA/VT-MAK_data/data_original/bombard_220725_006/bombard_220725_006_001/bombard_220725_006_220726_000000.csv', encoding="utf8") # Path
# reader = csv.reader(f)
# csv_list = []
# for l in reader:
#     csv_list.append(l)
# f.close()
# data_1 = pd.DataFrame(csv_list)
# print(data_1)

## 1개 csv 데이터 전처리
# data_1 = data_1.transpose() # transpose
# data_1.rename(columns=data_1.iloc[0], inplace=True)
# data_1 = data_1.drop(data_1.index[0])
# data_1 = data_1.loc[:, ['TimeStamp(sec)', 'EntityID', 'EntityType', 'Force', 'Location(GeoCentric X-Y-Z in (m))', 'Health']]
# data_1 = data_1.reset_index(drop=True)
# data_1 = data_1.rename(columns={'TimeStamp(sec)' : 'Time'}) # Time col 이름 변경
# # print(data_1)

# location = data_1['Location(GeoCentric X-Y-Z in (m))']
# data_1_preprocessed = data_1.drop(['Location(GeoCentric X-Y-Z in (m))'], axis=1) # location : (x,y,z)
# location = data_1['Location(GeoCentric X-Y-Z in (m))']
# location = location.drop(index=0) # x좌표 제거
# location = location.dropna()
# location = location.reset_index(drop=True)

# data_1_preprocessed['Location'] = location # 위치 col 이름 설정
# loc_y = location[0]
# loc_z = location[1]
# data_1_preprocessed['Location'][0] = loc_y + " " + loc_z # location을 하나의 row로 합침
# data_1_preprocessed = data_1_preprocessed.drop(index=1) # 필요없는 행 제거
# data_1_preprocessed = data_1_preprocessed.drop(index=2) # 필요없는 행 제거
# data_1_preprocessed = data_1_preprocessed[['Time', 'EntityID', 'EntityType', 'Force', 'Location', 'Health']] # col 순서 변경
# # print(data_1_preprocessed)
# # data_1_preprocessed.to_csv("data_1_preprocessed.csv", index=False) # 새로운 csv 파일 생성

#####################################################################

## 한 객체의 전체 csv 파일 읽기
# csv 파일 1개 -> 한 폴더(한 객체)로 확장
# 한 폴더의(한 객체) csv 파일 전체 -> 전처리 후 하나의 csv 파일로 합치고 time 정렬 후 저장

# files = os.listdir('./VT-MAK_DATA/VT-MAK_data/data_original') # Path - 전체 데이터 파일
# files = os.listdir('./Hyunseok/VT-MAK_DATA/VT-MAK_data/data_original')

# folders = 'VT-MAK_DATA/VT-MAK_data/data_original/bombard_220725_006/bombard_220725_006_001' # Path - 하나의 객체
# files_list = os.listdir(folders)
# # print(len(files_list)) # 263 (모든 파일들 list)
# csv_files = [file for file in files_list if file.endswith(".csv")] # 한 폴더의 csv 파일 전체
# # print(len(csv_files)) # 90 (csv 파일 list)

# one_entity_csv = pd.DataFrame()
# for csv_file in csv_files:
#     # 에러 처리
#     fs = open(folders+ '/' + csv_file)
#     readers = csv.reader(fs)
#     csvs_list = []
#     for ls in readers:
#         csvs_list.append(ls)
#     fs.close()
#     df = pd.DataFrame(csvs_list)

#     # 전처리
#     df = df.transpose() # transpose
#     df.rename(columns=df.iloc[0], inplace=True)
#     df = df.drop(df.index[0])
#     df = df.loc[:, ['TimeStamp(sec)', 'EntityID', 'EntityType', 'Force', 'Location(GeoCentric X-Y-Z in (m))', 'Health']]
#     df = df.reset_index(drop=True)
#     df = df.rename(columns={'TimeStamp(sec)' : 'Time'}) # Time col 이름 변경

#     df_preprocessed = df.drop(['Location(GeoCentric X-Y-Z in (m))'], axis=1) # location : (x,y,z)
#     location = df['Location(GeoCentric X-Y-Z in (m))']
#     location = location.drop(index=0) # x좌표 제거
#     location = location.dropna()
#     location = location.reset_index(drop=True)

#     df_preprocessed['Location'] = location # 위치 col 이름 설정
#     loc_y = location[0]
#     loc_z = location[1]
#     df_preprocessed['Location'][0] = loc_y + " " + loc_z # location을 하나의 row로 합침
#     df_preprocessed = df_preprocessed.drop(index=1) # 필요없는 행 제거
#     df_preprocessed = df_preprocessed.drop(index=2) # 필요없는 행 제거
#     df_preprocessed = df_preprocessed[['Time', 'EntityID', 'EntityType', 'Force', 'Location', 'Health']] # col 순서 변경
#     df_preprocessed['Time'] = float(df_preprocessed['Time']) # time col 처리

#     one_entity_csv = pd.concat([one_entity_csv, df_preprocessed]) # 객체 별로 하나의 csv 파일로 합침

# one_entity_csv = one_entity_csv.sort_values(by=['Time']) # time 순으로 정렬
# one_entity_csv = one_entity_csv.reset_index(drop=True) # index 재설정
# # one_entity_csv.to_csv(folders + "/{}.csv".format(one_entity_csv['EntityID']), index=False) # 하나의 entity csv 파일 생성
# # print(one_entity_csv) # 90x6

#####################################################################

## 시나리오 폴더 단위 읽기, 전처리를 함수 형태로 작성
# 한 객체 단위 -> 한 시나리오 단위로 확장(시나리오의 각 객체별로 csv 파일 저장)
def data_preprocess_scenario(path, scenario):
    print(scenario)
    path = path+scenario+'/'# path - 한 시나리오 폴더(19 객체 전부)
    outpath = './VT-MAK_DATA/VT-MAK_data/data_preprocessed/' # 전처리 된 csv가 저장될 위치 (객체별 저장)
    folders = os.listdir(path)
    out_folders = os.listdir(outpath)
    os.makedirs(outpath+scenario, exist_ok=True)
    # print(folders)

    #one_entity_csv = pd.DataFrame() # 객체별 csv 통합 파일 저장을 위해 정의
    entity_list = pd.DataFrame() # 객체별 csv 통합 파일 저장을 위해 정의

    # def DataPreprocess(folders): # 함수 형태
    for folder in folders: # 한 시나리오의 19 객체 loop
        one_entity_csv = pd.DataFrame(columns=['Time', 'EntityID', 'EntityType', 'Force', 'Location_y', 'Location_z', 'Health']) # 객체별 csv 통합 파일 저장을 위해 정의
        folderName = folder # 각 폴더명 저장

        files = os.listdir(path + folder) # 불러온 파일들을 리스트에 저장
        files = [file_csv for file_csv in files if file_csv.endswith(".csv")] # csv 파일만 불러옴
        idx = 0
        for file in files: # csv 파일 단위(90초) loop
            fileName = file # 각 파일명 저장
            all_csv_files = path + folderName + '/' + fileName # 폴더와 파일 이름 조합
            ######  #EntityStatePDU 누락 데이터 있을 시 아래 소스 주석 해제해서 파일 확인

            print(folderName)
            print(file)
            
            
            # 에러 처리
            f = open(all_csv_files)
            reader = csv.reader(f)
            csv_list = []
            for l in reader:
                csv_list.append(l)
            f.close() 
            
            # 전처리
            data_raw = pd.DataFrame(csv_list)
            data_raw = data_raw.loc[:31, :] # detection information 제거
            data_raw = data_raw.transpose() # transpose
    # print(df)
            data_raw.rename(columns=data_raw.iloc[0], inplace=True) # col 이름 설정
            data_raw = data_raw.drop(data_raw.index[0])
            print(folder)
            entity_type = data_raw['EntityType'].iloc[0]
            
            data_raw['EntityType'] = str(entity_type).split(':')[0] #+":" + str(entity_type).split(':')[1]
            if data_raw['EntityType'].iloc[0] == "3" :
                data_raw = data_raw.loc[:, ['TimeStamp(sec)', 'EntityID', 'EntityType', 'Force', 'Location(GeoCentric X-Y-Z in (m))', 'Health']]
            elif data_raw['EntityType'].iloc[0] == "1" :
                data_raw = data_raw.loc[:, ['TimeStamp(sec)', 'EntityID', 'EntityType', 'Force', 'Location(GeoCentric X-Y-Z in (m))', 'Damage']]
                data_raw = data_raw.rename(columns={'Damage' : 'Health'}) # Time col 이름 변경
            data_raw = data_raw.reset_index(drop=True) # 인덱스 0부터 되게 초기화
            data_raw = data_raw.rename(columns={'TimeStamp(sec)' : 'Time'}) # Time col 이름 변경

    # print(df)
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
            df_preprocessed['Time'] = float(df_preprocessed['Time']) # time col 값 실수로 처리

            one_entity_csv = pd.concat([one_entity_csv, df_preprocessed])
            
            #print(df_preprocessed)        
        
        one_entity_csv = one_entity_csv.sort_values(by=['Time']) # time 순으로 정렬
        one_entity_csv = one_entity_csv.reset_index(drop=True) # index 재설정    
        one_entity_csv.to_csv(outpath+scenario+'/'+folder+".csv",index=False,encoding="utf-8-sig")
        #print(one_entity_csv)    
            # entityId = df_preprocessed['EntityID']# 객체 별로 csv 나눔 (19개 size : len(time) x 6)
            # max_time = max(df_preprocessed['Time'])
            # # df_preprocessed = df_preprocessed.sort_values(df_preprocessed['Time']) # 나눈 후 time 순으로 정렬
            # # df_preprocessed = df_preprocessed.reset_index(drop=True) # index 재설정
            
            # ## entity id가 달라지면 나눔 or Time이 max가 될 때 나눔
            # if float(df_preprocessed['Time']) == max_time:
            #     entity_list.append(files)
            #     one_entity_csv = pd.concat( entity_list) # to dataframe
            #     entity_list = [] # initialize
            
        ###########################
        # one_entity_csv = pd.concat(one_entity_csv) # 객체 별로 하나의 csv 파일로 합침
    #one_entity_csv = one_entity_csv.sort_values(by=df_preprocessed['Time']) # 나눈 후 time 순으로 정렬
    #one_entity_csv = one_entity_csv.reset_index(drop=True) # index 재설정

            # idx += 1 # 인덱스
        
            # print(one_entity_csv)
    #print(one_entity_csv)
            
    # print(len(files)) # 90 x 19 (중간에 89 섞여 있음)

        # print(file)
        # print(folders)
    
    
    # one_entity_csv = pd.concat([one_entity_csv, df_preprocessed]) # 객체 별로 하나의 csv 파일로 합침
    # one_entity_csv = one_entity_csv.sort_values(by=['Time']) # time 순으로 정렬
    # one_entity_csv = one_entity_csv.reset_index(drop=True) # index 재설정
    
### EntityType 1:2:225:3:2:0:0 데이터(006_003)에 Health 정보 없음(대신 Demage 정보가 있음) ###
    
    # one_entity_csv.to_csv(out_folders + "/{}.csv".format(one_entity_csv['EntityID']), index=False) # 객체(entity) 별로 csv 파일 생성

    # print('\n', df_preprocessed)
    # print('\n', one_entity_csv)
    
# print(df_preprocessed)

# print(len(all_csv_files)) # 116


#####################################################################

## 2D mapping - (y,z)? -> max, min
# map = [[]]
# max_y = 
# min_y = 
# max_z = 
# min_z = 
# coord_2d = 
def preprocessing_main(args):
    path = args.path
    folders = os.listdir(path)
    for scenario in folders:
        data_preprocess_scenario(path,scenario)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Recommend Model')
    parser.add_argument('--path', type=str, default='./VT-MAK_DATA/VT-MAK_data/data_original/')


    args = parser.parse_args()
    preprocessing_main(args)