# -*- coding: utf-8 -*-
"""
battlefiled awareness model training module
"""
import argparse
import os
import random
import time
import torch
import numpy as np
import pandas as pd
from convlstm import ConvLSTM

torch.manual_seed(7)  # random seed 잡아주기
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("using device :", DEVICE)

CHANNELS = 4
WIDTH = 64
HEIGHT = 64
BATCH_TIME = 16
DAMAGE_TIME = 5
MAX_TIMESTAMP = 151
DV_TIME = 5
DAMAGE_CLASS_NUM = 5


def acc_check(net, test_x, test_y):
    """
    accuracy check function
    Args:
        net: nn.modules
            convlstm model
        test_x: tensor
            test data x
        test_y: tensor
            test data y

    Returns:
        acc : float (winrate, friendly damage, opponent damage)
    """
    with torch.no_grad():

        predict_y = net(test_x)
        correct_wr = sum((test_y[:, 0] - predict_y[:, 0]) <= 0.5) / len(test_y)

        predicted_fd = torch.max(predict_y[:, 1:6], 1)
        y_fd = torch.max(test_y[:, 1:6], 1)
        correct_fd = (predicted_fd.indices == y_fd.indices).float().mean()

        predicted_od = torch.max(predict_y[:, 6:11], 1)
        y_od = torch.max(test_y[:, 6:11], 1)
        correct_od = (predicted_od.indices == y_od.indices).float().mean()

    acc_wr = 100 * correct_wr
    acc_fd = 100 * correct_fd
    acc_od = 100 * correct_od
    return acc_wr, acc_fd, acc_od


def main(path):
    """
    main function - training model
    Args:
        path : str
            data path
    Returns:
        training time, loss, acc(wr,fd,od) : float
    """
    data_x, data_y = data_generating(path)
    weight_path = "./weights/"

    test_num = int(len(data_y) * 0.1)
    test_idx = random.sample(range(0, len(data_y)), test_num)
    x = np.zeros((len(data_y) - test_num, BATCH_TIME, CHANNELS, WIDTH, HEIGHT))
    y = []
    tx = np.zeros((test_num, BATCH_TIME, CHANNELS, WIDTH, HEIGHT))
    ty = []
    train_num = 0
    test_num = 0
    for i in range(len(data_y)):
        if i in test_idx:
            tx[test_num] = data_x[i].copy()
            ty.append(data_y[i])
            test_num += 1
        else:
            x[train_num] = data_x[i].copy()
            y.append(data_y[i])
            train_num += 1

    train_x = torch.Tensor(x).to(DEVICE)
    train_y = torch.Tensor(y).to(DEVICE)
    test_x = torch.Tensor(tx).to(DEVICE)
    test_y = torch.Tensor(ty).to(DEVICE)

    model = ConvLSTM(input_dim=CHANNELS,
                     hidden_dim=6,
                     kernel_size=(3, 3),
                     num_layers=1,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False).to(DEVICE)

    criterion = torch.nn.BCELoss().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, betas=(
            0.9, 0.999), weight_decay=0.0001)

    start_time = time.time()
    print("train start")
    acc_temp = 0
    step = 0
    while True:
        model.train()
        optimizer.zero_grad()
        hypothesis = model(train_x)
        loss = criterion(hypothesis, train_y)
        loss.backward()

        optimizer.step()
        model.eval()
        acc_wr, acc_fd, acc_od = acc_check(model, test_x, test_y)

        if (acc_fd > acc_temp):
            torch.save(model, weight_path + 'model.pt')
            save_step = step
            save_loss = loss.item()
            save_acc_wr = acc_wr.item()
            save_acc_fd = acc_fd.item()
            save_acc_od = acc_od.item()
        if step > 50 and acc_fd <= acc_temp:
            # 종료조건 설정
            break
        if step % 10 == 0:
            print(
                "step : ",
                step,
                ", loss : ",
                loss.item(),
                " acc : ",
                acc_wr.item(),
                acc_fd.item(),
                acc_od.item())
        acc_temp = acc_fd
        step += 1
    print("train finished")
    training_time = time.time() - start_time
    print(f"Time: {training_time:.5f}sec")
    print("best model training step : ", save_step)
    print("best model loss : ", save_loss)
    print(
        "best model acc : ",
        save_acc_wr,
        "\t",
        save_acc_fd,
        "\t",
        save_acc_od)
    return training_time, save_loss, save_acc_wr, save_acc_fd, save_acc_od

# 각 시나리오 별로 최대 좌표값을 직접 찾아서 2D map size 저장
def find_map_size(folders):
    """
    map size check function
    Args:
        folders: list
            scenario folder list
    Returns:
        mapsize : float
        move coord : float
    """

    mapsize_y = []
    mapsize_z = []
    mapsize_y_for_zero = []
    mapsize_z_for_zero = []
    print('Scenario folder list')
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


# 좌표 값 정규화
def coordinateTransform(path, folders):
    """
    coordinate transform function
    Args:
        path: str
            data path
        folders: list
            scenario folder list

    Returns:
        save csv file
    """
    mapsize_y, mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero = \
        find_map_size(folders)
    scenario_num = 0
    tmppath = './tmp/'  # 임시 csv 파일 위치 (객체별 저장)

    for scenario in folders:
        data_folder = os.listdir(path + scenario)
        os.makedirs(tmppath + scenario, exist_ok=True)
        for data_file in data_folder:
            data = pd.read_csv(path + scenario + '/' + data_file, sep=",")
            data['Location_y'] = data['Location_y'].astype('float')
            data['Location_z'] = data['Location_z'].astype('float')

            data['Location_y'] = data['Location_y'] - \
                int(mapsize_y_for_zero[scenario_num])
            data['Location_z'] = data['Location_z'] - \
                int(mapsize_z_for_zero[scenario_num])
            data['Location_y'] = data['Location_y'] * \
                (WIDTH - 1) / int(mapsize_y[scenario_num])
            data['Location_z'] = data['Location_z'] * \
                (HEIGHT - 1) / int(mapsize_z[scenario_num])

            data.to_csv(
                tmppath +
                scenario +
                '/' +
                data_file,
                index=False,
                encoding="utf-8-sig")
        scenario_num += 1


# 모든 시나리오에 대하여, 정해진 W,H,C,T 값으로 2d맵 생성 및 데이터 업데이트
def makeCoord(folders):
    """
    making map data function
    Args:
        folders: list
            scenario folder list
    Returns:
        coord_data : np.array
            map data
        scenario_time : list
            scenario time stamp list
    """
    path = './tmp/'
    scenario_num = 0
    scenario_time = []
    # scenario, time, unit(channel), w, h
    # 전체 맵 생성
    coord_data = np.zeros(
        (len(folders), MAX_TIMESTAMP, CHANNELS, WIDTH, HEIGHT))
    for scenario in folders:
        data_folder = os.listdir(path + scenario)
        os.makedirs(path + scenario, exist_ok=True)
        for data_file in data_folder:
            data = pd.read_csv(path + scenario + '/' + data_file, sep=",")
            ts = data['Time'].max()
            for t in data['Time']:

                if t % 1 == 0:
                    idx_num = data.index[(data['Time'] == t)]
                    # 유닛이 사망하지 않았을 경우
                    if str(
                            data['Health'].iloc[idx_num].iloc[0]) !=\
                            'Destroyed':
                        # 예외처리
                        if int(
                                data['Location_y'].iloc[idx_num]) <=\
                                WIDTH - 1 and int(
                                data['Location_y'].iloc[idx_num]) >=\
                                0 and int(
                                data['Location_z'].iloc[idx_num]) <=\
                                HEIGHT - 1 and int(
                                data['Location_z'].iloc[idx_num]) >=\
                                0:
                            # 아군,적군에 따라 생성된 맵에 업데이트
                            if data['Force'].iloc[0] == 'Friendly':

                                # 0번 채널 아군 human 1번 채널 아군 탈것 2번채널 적군 human
                                # 3번채널 적군 탈것
                                if data['EntityType'].iloc[0] == 3:
                                    coord_data[scenario_num][int(t)][0][int(
                                        data['Location_y'].iloc[idx_num])][int(
                                            data['Location_z'].iloc[idx_num])]\
                                        += 1
                                else:
                                    coord_data[scenario_num][int(t)][1][int(
                                        data['Location_y'].iloc[idx_num])][int(
                                            data['Location_z'].iloc[idx_num])]\
                                        += 1
                            else:
                                if data['EntityType'].iloc[0] == 3:
                                    coord_data[scenario_num][int(t)][2][int(
                                        data['Location_y'].iloc[idx_num])][int(
                                            data['Location_z'].iloc[idx_num])]\
                                        += 1
                                else:
                                    coord_data[scenario_num][int(t)][3][int(
                                        data['Location_y'].iloc[idx_num])][int(
                                            data['Location_z'].iloc[idx_num])]\
                                        += 1

        scenario_num += 1

        scenario_time.append(int(ts))
    return coord_data, scenario_time

# 
def output_data_tagging(coord_data, scenario_time):
    """
    label tagging function
    Args:
        coord_data: np.array
            map data
        scenario_time : list
            scenario time stamp list
    Returns:
        health_fren, health_oppo : list
            health state
        is_win : list
            whether win or not
    """
    health_fren = np.zeros((len(scenario_time), MAX_TIMESTAMP)) # 시간에 따른 아군, 적군의 체력 상태를 저장하는 리스트
    health_oppo = np.zeros((len(scenario_time), MAX_TIMESTAMP))
    is_win = [] # 시나리오의 승리 여부

    for i in range(len(scenario_time)):
        # 맵 데이터의 체력 데이터를 합산해 해당 위치에 할당
        for j in range(scenario_time[i]):
            # i는 시나리오 번호, j는 시나리오의 타임스탬프 (최대는 타임길이)
            health_fren[i][j] = np.sum(
                coord_data[i][j][0]) + np.sum(coord_data[i][j][1])
            health_oppo[i][j] = np.sum(
                coord_data[i][j][2]) + np.sum(coord_data[i][j][3])
        # 아군, 적군 체력 비교해서 1, 0.5, 0으로 나눔
        if health_fren[i][scenario_time[i] -
                          1] > health_oppo[i][scenario_time[i] - 1]:
            is_win.append(1)
        elif health_fren[i][scenario_time[i] - 1] \
                < health_oppo[i][scenario_time[i] - 1]:
            is_win.append(0)
        else:
            is_win.append(0.5)
    return health_fren, health_oppo, is_win

# 피해 정도를 5단계로 나눔
def damage_to_classification(damageRate):
    """
    damage classification function
    Args:
        damageRate: int

    Returns:
        tmp : list
            damage classㅋ
    """
    if damageRate < 0:
        damageRate = 0
    damage_class = damageRate * DAMAGE_CLASS_NUM
    tmp = []
    for i in range(DAMAGE_CLASS_NUM):
        if i == int(damage_class):
            tmp.append(1)
        elif i == DAMAGE_CLASS_NUM - 1 \
                and int(damage_class) >= DAMAGE_CLASS_NUM - 1:
            tmp.append(1)
        else:
            tmp.append(0)
    return tmp

# input data를 5초 간격으로 슬라이딩해서 train data와 train label로 변환
def data_sliding(coord_data, scenario_time, health_fren, health_oppo, is_win):
    """
    data slicing
    Args:
        coord_data: np.array
            map data
        scenario_time: list
            scenario time list
        health_fren: friendly damage
        health_oppo: opponent damage
        is_win: whether win or not

    Returns:
        data_x : train data
        data_y : train label
    """
    total_timestamp = sum(scenario_time) + len(scenario_time) # 시나리오의 총 time stamp 개수
    scenario_time_dv5 = (scenario_time / DV_TIME) # 시나리오의 time을 5로 나눈 값
    for i in range(len(scenario_time_dv5)):

        if scenario_time_dv5[i] % 1 != 0:
            scenario_time_dv5[i] = scenario_time_dv5[i] + (1 / DV_TIME)
            scenario_time_dv5[i] = scenario_time_dv5[i] // 1

    total_batch = int(sum(scenario_time_dv5)) # 총 배치 수를 시나리오 time을 5로 나눈 값을 정수로 합해서 구함
    data_x = np.zeros((total_batch, BATCH_TIME, CHANNELS, WIDTH, HEIGHT)) # train data  배열 초기화, 크기를 5차원으로 함, 모델에 input 하기 위해
    data_y_wr = []
    data_y_fd = []
    data_y_od = []
    scenario_pointer = 0
    time_pointer = 0
    batch_pointer = 0
    # 반복문 돌면서() 데이터 슬라이딩, 레이블 생성
    for i in range(total_timestamp):
        time_pointer += 1
        if int(scenario_time[scenario_pointer]) + 1 == time_pointer:
            time_pointer = 0
            scenario_pointer += 1
        if time_pointer % DV_TIME == DV_TIME - 1:
            # 
            for t in range(BATCH_TIME):
                if time_pointer - BATCH_TIME + t >= 0:
                    data_x[batch_pointer][t] = coord_data[
                        scenario_pointer][time_pointer - BATCH_TIME + t].copy()
            wr = 0.5 + ((is_win[scenario_pointer] - 0.5) *
                        time_pointer / scenario_time[scenario_pointer])
            data_y_wr.append(wr)
            if health_fren[scenario_pointer][time_pointer] == 0:
                tmp = damage_to_classification(0)
            else:
                if time_pointer + DAMAGE_TIME - \
                        1 <= scenario_time[scenario_pointer]:
                    tmp = damage_to_classification(
                        (health_fren[scenario_pointer][time_pointer] -
                         health_fren[scenario_pointer][
                            time_pointer +
                            DAMAGE_TIME -
                            1]) /
                        health_fren[scenario_pointer][time_pointer])
                else:
                    tmp = damage_to_classification((
                        health_fren[scenario_pointer][time_pointer] -
                        health_fren[scenario_pointer]
                        [scenario_time[scenario_pointer]]) /
                        health_fren[scenario_pointer][time_pointer])
            data_y_fd.append(tmp)
            if health_oppo[scenario_pointer][time_pointer] == 0:
                tmp = damage_to_classification(0)
            else:
                if time_pointer + DAMAGE_TIME - \
                        1 <= scenario_time[scenario_pointer]:
                    tmp = damage_to_classification(
                        (health_oppo[scenario_pointer][time_pointer] -
                         health_oppo[scenario_pointer][
                            time_pointer +
                            DAMAGE_TIME -
                            1]) /
                        health_oppo[scenario_pointer][time_pointer])
                else:
                    tmp = damage_to_classification((
                        health_oppo[scenario_pointer][time_pointer] -
                        health_oppo[scenario_pointer]
                        [scenario_time[scenario_pointer]]) /
                        health_oppo[scenario_pointer][time_pointer])
            data_y_od.append(tmp)
            batch_pointer += 1

    data_y = np.zeros((total_batch, 1 + len(data_y_fd[0]) + len(data_y_od[0]))) # label data 배열 초기화, 크기 설정
    for i in range(total_batch):
        data_y[i][0] = data_y_wr[i]
        for j in range(len(data_y_fd[0])):
            data_y[i][j + 1] = data_y_fd[i][j]
        for j in range(len(data_y_od[0])):
            data_y[i][j + 1 + len(data_y_fd[0])] = data_y_od[i][j]

    return data_x, data_y

# data augmentation
def data_aug(data_x, scenario_time):
    """
    augmentation function
    Args:
        data_x: np.array
            train data
        scenario_time: list
            scenario timestamp list

    Returns:
        data_x : np.array
            augmented data
        aug_scenario_time : list
            augmented scenario timestamp list
    """
    aug_data_x = np.zeros(data_x.shape)  # scenario, time, unit(channel), w, h
    aug_scenario_time = np.concatenate([scenario_time, scenario_time])
    for i in range(len(data_x)):
        for j in range(scenario_time[i]):
            aug_data_x[i][j][0] = data_x[i][j][2].copy()
            aug_data_x[i][j][1] = data_x[i][j][3].copy()
            aug_data_x[i][j][2] = data_x[i][j][0].copy()
            aug_data_x[i][j][3] = data_x[i][j][1].copy()
    data_x = np.concatenate([data_x, aug_data_x])

    return data_x, aug_scenario_time

# 데이터 path에서 데이터를 생성하고 처리하여 train data, label로 리턴
def data_generating(path):
    """
    from scenario data to training data function
    Args:
        path: str
            data path

    Returns:
        data_x : np.array
            train data
        data_y : list
            train label
    """
    folders = os.listdir(path)
    print("Data generation start(scenario -> batch)")
    coordinateTransform(path, folders)
    coord_data, scenario_time = makeCoord(folders) # 시나리오로 부터, 2d map 데이터 생성하기 (생성된 2d map,  시나리오 시간)

    print("Data augmentation start")
    coord_data, scenario_time = data_aug(coord_data, scenario_time)
    print("Data augmentation finished")
    health_fren, health_oppo, is_win = output_data_tagging(
        coord_data, scenario_time)
    data_x, data_y = data_sliding(
        coord_data, scenario_time, health_fren, health_oppo, is_win)
    print("Data generation finished")
    print("total batch : ", len(data_x))
    return data_x, data_y


if __name__ == "__main__":  
    parser = argparse.ArgumentParser('Situation awareness Model')
    parser.add_argument(
        '--path',
        type=str,
        default="./VT_MAK_data/data_preprocessed/")
    args = parser.parse_args()
    main(args.path)
    # data_generating(args.path)


    # x, y = data_generating(args.path)

    # print("x값 확인 : ", x)
    # print("y값 확인 : ", y)
    # print("x shape 확인 : ", x.shape) # (242, 30, 4, 64, 64) - total batch : 242
    # print("y shape 확인 : ", y.shape) # (242, 11)
    