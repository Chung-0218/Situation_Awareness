# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
from convlstm import ConvLSTM
import argparse
import os
import random

torch.manual_seed(7) #####random seed 잡아주기
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device :",device)

channels = 4
width = 64
height = 64
time = 30
max_timestamp = 151
def main(args):
    weight_path = "./weights/"

    data_x, data_ts = data_generating_predict(args.path)
    x = torch.Tensor(data_x).to(device)
    
    model = torch.load(weight_path + args.weights+'model.pt')
    
    #model.load_state_dict(torch.load(PATH + args.weights + 'model_state_dict.pt'))
    with torch.no_grad():
        predicted = model(x)
        df_y = pd.DataFrame(predicted.detach().cpu().numpy(),columns = ['승률','매우경미(아)','경미(아)','중간(아)','심각(아)','매우심각(아)','매우경미(적)','경미(적)','중간(적)','심각(적)','매우심각(적)'])

        output = pd.concat([data_ts,df_y],axis=1)

        output.sort_values(by=['scenario_name','timestamp'],axis=0,ascending=True,inplace= True) ## 정렬을 원할 경우 by

        output.to_csv("./output/predict.csv",mode="w",index=False)
        print(output, '\n')


def find_map_size(path, folders):
    # #이상치 있을 경우 좌표 체크 코드
    # for scenario in folders:
    #     data_folder = os.listdir(path+scenario)
    #     for data_file in data_folder:
    #         data = pd.read_csv(path+scenario+'/'+data_file,sep=",")
    #         if scenario.startswith("lure") :
    #             print(data['Location_y'].astype('float').max())
    #             print(data['Location_y'].astype('float').min())

    mapsize_y = []
    mapsize_z = []
    mapsize_y_for_zero = []
    mapsize_z_for_zero = []
    for scenario in folders:
        print(scenario)
        if scenario.startswith("lure"):
            mapsize_y.append(225)
            mapsize_z.append(224)
            mapsize_y_for_zero.append(-113)
            mapsize_z_for_zero.append(-111)
        elif scenario.startswith("bridge"):
            mapsize_y.append(225)
            mapsize_z.append(224)
            mapsize_y_for_zero.append(-113)
            mapsize_z_for_zero.append(-111)
        elif scenario.startswith("delay"):
            mapsize_y.append(418)
            mapsize_z.append(365)
            mapsize_y_for_zero.append(-4332732)
            mapsize_z_for_zero.append(3729141)
        elif scenario.startswith("hall"):
            mapsize_y.append(117)
            mapsize_z.append(191)
            mapsize_y_for_zero.append(-4264607)
            mapsize_z_for_zero.append(3883851)
        elif scenario.startswith("bombard"):
            mapsize_y.append(250)
            mapsize_z.append(250)
            mapsize_y_for_zero.append(-125)
            mapsize_z_for_zero.append(-125)
    return mapsize_y, mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero


def coordinateTransform(path, folders):
    mapsize_y, mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero = find_map_size(path, folders)
    scenario_num = 0
    tmppath = './tmp/'  # 임시 csv 파일 위치 (객체별 저장)

    # print(mapsize_y,mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero)
    for scenario in folders:
        data_folder = os.listdir(path + scenario)
        os.makedirs(tmppath + scenario, exist_ok=True)
        for data_file in data_folder:
            data = pd.read_csv(path + scenario + '/' + data_file, sep=",")
            data['Location_y'] = data['Location_y'].astype('float')
            data['Location_z'] = data['Location_z'].astype('float')

            data['Location_y'] = data['Location_y'] - int(mapsize_y_for_zero[scenario_num])
            data['Location_z'] = data['Location_z'] - int(mapsize_z_for_zero[scenario_num])
            data['Location_y'] = data['Location_y'] * (width - 1) / int(mapsize_y[scenario_num])
            data['Location_z'] = data['Location_z'] * (height - 1) / int(mapsize_z[scenario_num])

            data.to_csv(tmppath + scenario + '/' + data_file, index=False, encoding="utf-8-sig")
        scenario_num += 1


def makeCoord_predict(folders):
    path = './tmp/'
    scenario_num = 0
    scenario_name = []
    scenario_time = []
    coord_data = np.zeros((len(folders), max_timestamp, channels, width, height))  # scenario, time, unit(channel), w, h
    for scenario in folders:
        data_folder = os.listdir(path + scenario)
        os.makedirs(path + scenario, exist_ok=True)
        for data_file in data_folder:
            data = pd.read_csv(path + scenario + '/' + data_file, sep=",")
            ts = data['Time'].max()
            for t in data['Time']:

                if t % 1 == 0:
                    idx_num = data.index[(data['Time'] == t)]
                    if str(data['Health'].iloc[idx_num].iloc[0]) != 'Destroyed':
                        if int(data['Location_y'].iloc[idx_num]) <= width - 1 and int(
                                data['Location_y'].iloc[idx_num]) >= 0 and int(
                                data['Location_z'].iloc[idx_num]) <= height - 1 and int(
                                data['Location_z'].iloc[idx_num]) >= 0:
                            if data['Force'].iloc[0] == 'Friendly':

                                if data['EntityType'].iloc[
                                    0] == 3:  # 0번 채널 아군 human 1번 채널 아군 탈것 2번채널 적군 human  3번채널 적군 탈것
                                    coord_data[scenario_num][int(t)][0][int(data['Location_y'].iloc[idx_num])][
                                        int(data['Location_z'].iloc[idx_num])] += 1
                                else:
                                    coord_data[scenario_num][int(t)][1][int(data['Location_y'].iloc[idx_num])][
                                        int(data['Location_z'].iloc[idx_num])] += 1
                            else:
                                if data['EntityType'].iloc[0] == 3:
                                    coord_data[scenario_num][int(t)][2][int(data['Location_y'].iloc[idx_num])][
                                        int(data['Location_z'].iloc[idx_num])] += 1
                                else:
                                    coord_data[scenario_num][int(t)][3][int(data['Location_y'].iloc[idx_num])][
                                        int(data['Location_z'].iloc[idx_num])] += 1

        scenario_num += 1
        scenario_name.append(scenario)
        scenario_time.append(int(ts))
    scenario_ts = pd.DataFrame({ 'scenario_name' : scenario_name, 'timestamp':scenario_time})
    return coord_data, scenario_ts


def data_sliding_predict(coord_data, scenario_ts):
    total_batch = sum(scenario_ts['timestamp'])
    data_x = np.zeros((total_batch, time, channels, width, height))
    data_scenario = []
    data_timestamp = []
    scenario_pointer = 0
    time_pointer = 0
    for i in range(total_batch):
        for t in range(time):
            if time_pointer-time +t >=0:
                data_x[i][t] = coord_data[scenario_pointer][time_pointer -time + t].copy()

        data_scenario.append(scenario_ts['scenario_name'].iloc[scenario_pointer])
        data_timestamp.append(time_pointer)

        time_pointer += 1
        if int(scenario_ts['timestamp'].iloc[scenario_pointer]) == time_pointer :
            time_pointer = 0
            scenario_pointer += 1

    data_ts = pd.DataFrame({'scenario_name': data_scenario, 'timestamp': data_timestamp})

    return data_x, data_ts


def data_generating_predict(path):
    folders = os.listdir(path)

    coordinateTransform(path, folders)
    coord_data, scenario_ts = makeCoord_predict(folders)
    data_x, data_ts = data_sliding_predict(coord_data, scenario_ts) #ts는 시나리오명과 timestamp가 포함된 data frame
    return data_x, data_ts
    """
    for scenario in folders:
        data_folder = os.listdir(path+scenario)
        for data_file in data_folder:
            data = pd.read_csv(path+scenario+'/'+data_file,sep=",")
    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Recommend Model')
    parser.add_argument('--weights', type=str, default="")
    parser.add_argument('--path', type=str, default="./VT_MAK_data/predict/")

    args = parser.parse_args()
    main(args)