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
damage_time = 10
max_timestamp = 151
def acc_check(net,test_x,test_y):

    
    with torch.no_grad():
        
        predict_y = net(test_x)
        correct_wr = sum((test_y[:,0] - predict_y[:,0]) <= 0.5)/len(test_y)

        predicted_fd = torch.max(predict_y[:,1:6],1)
        y_fd= torch.max(test_y[:,1:6],1)
        correct_fd = (predicted_fd.indices == y_fd.indices ).float().mean()

        predicted_od = torch.max(predict_y[:,6:11],1)
        y_od= torch.max(test_y[:,6:11],1)
        correct_od = (predicted_od.indices == y_od.indices ).float().mean()

    acc_wr= 100*correct_wr
    acc_fd= 100*correct_fd
    acc_od= 100*correct_od
    #print('Accuracy of the network on the test : ', acc_wr.item()," %", acc_fd.item()," %", acc_od.item()," %")
    return acc_wr, acc_fd, acc_od
def main(args):
    data_x, data_y = data_generating(args.path)
    weight_path = "./weights/"
    
    test_num = int(len(data_y) * 0.1)
    test_idx = random.sample(range(0,len(data_y)),test_num)
    x = np.zeros((len(data_y)-test_num,time,channels,width,height))
    y = []
    tx = np.zeros((test_num,time,channels,width,height))
    ty = []
    train_num =0
    test_num = 0
    for i in range(len(data_y)):
        if i in test_idx:
            tx[test_num] = data_x[i].copy()
            ty.append(data_y[i])
            test_num+=1
        else :
            x[train_num] = data_x[i].copy()
            y.append(data_y[i])
            train_num+=1
        
            
    

    train_x = torch.Tensor(x).to(device)
    train_y = torch.Tensor(y).to(device)
    test_x = torch.Tensor(tx).to(device)
    test_y = torch.Tensor(ty).to(device)

    
    
    #model
    
    model = ConvLSTM(input_dim=channels,
                 hidden_dim=6,
                 kernel_size=(3, 3),
                 num_layers=1,
                 batch_first=True,
                 bias=True,
                 return_all_layers=False).to(device)

   
    #define cost/loss&optimizer
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9,0.999),weight_decay=0.0001)
    #lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    print("train start")
    acc_temp=0
    step = 0 
    while True:
        model.train()
        optimizer.zero_grad()
        hypothesis = model(train_x)
        #lr_sche.step()
        ##cost/loss function

        loss = criterion(hypothesis, train_y)
        loss.backward()
        # print(hypothesis)
        # print(train_y)

        optimizer.step()
        model.eval()
        acc_wr, acc_fd, acc_od = acc_check(model,test_x,test_y)
        
        if(acc_fd > acc_temp) :
                torch.save(model, weight_path + args.weights +'model.pt')
                save_step = step
                save_loss = loss.item()
                save_acc_wr = acc_wr.item()
                save_acc_fd = acc_fd.item()
                save_acc_od = acc_od.item()
                #torch.save(model.state_dict(), PATH + args.weights +'model_state_dict.pt')
        if step > 150 and acc_fd <=acc_temp :
            #종료조건 설정
            break
        if step % 10 ==0:
            print("step : ",step,", loss : ", loss.item()," acc : ", acc_wr.item(), acc_fd.item(), acc_od.item())
        acc_temp = acc_fd
        step +=1
    print("train finished")
    print("best model training step : ", save_step)
    print("best model loss : ",save_loss)
    print("best model acc : ", save_acc_wr, "\t", save_acc_fd, "\t", save_acc_od)


def find_map_size(path,folders):
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
    print('Scenario folder list')
    for scenario in folders:
        print(scenario)
        if scenario.startswith("lure") : 
            mapsize_y.append(225)
            mapsize_z.append(224)
            mapsize_y_for_zero.append(-113)
            mapsize_z_for_zero.append(-111)
        elif scenario.startswith("bridge") : 
            mapsize_y.append(225)
            mapsize_z.append(224)
            mapsize_y_for_zero.append(-113)
            mapsize_z_for_zero.append(-111)
        elif scenario.startswith("delay") : 
            mapsize_y.append(418)
            mapsize_z.append(365)
            mapsize_y_for_zero.append(-4332732)
            mapsize_z_for_zero.append(3729141)
        elif scenario.startswith("hall") : 
            mapsize_y.append(117)
            mapsize_z.append(191)
            mapsize_y_for_zero.append(-4264607)
            mapsize_z_for_zero.append(3883851)
        elif scenario.startswith("bombard") : 
            mapsize_y.append(250)
            mapsize_z.append(250)
            mapsize_y_for_zero.append(-125)
            mapsize_z_for_zero.append(-125)
    return mapsize_y,mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero
def coordinateTransform(path, folders):
    mapsize_y,mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero = find_map_size(path,folders)
    scenario_num=0
    tmppath = './tmp/' # 임시 csv 파일 위치 (객체별 저장)
    
  
    #print(mapsize_y,mapsize_z, mapsize_y_for_zero, mapsize_z_for_zero)
    for scenario in folders:
        data_folder = os.listdir(path+scenario)
        os.makedirs(tmppath+scenario, exist_ok=True)
        for data_file in data_folder:
            data = pd.read_csv(path+scenario+'/'+data_file,sep=",")
            data['Location_y'] = data['Location_y'].astype('float')
            data['Location_z'] = data['Location_z'].astype('float')

            data['Location_y'] = data['Location_y'] - int(mapsize_y_for_zero[scenario_num])
            data['Location_z'] = data['Location_z'] - int(mapsize_z_for_zero[scenario_num])
            data['Location_y'] = data['Location_y'] * (width-1)/int(mapsize_y[scenario_num])
            data['Location_z'] = data['Location_z'] * (height-1)/int(mapsize_z[scenario_num])
            
            data.to_csv(tmppath+scenario+'/'+data_file,index=False,encoding="utf-8-sig")
        scenario_num+=1
def makeCoord(folders):
    path = './tmp/' 
    scenario_num=0
    scenario_time=[]
    coord_data = np.zeros((len(folders),max_timestamp,channels,width,height)) #scenario, time, unit(channel), w, h
    for scenario in folders:
        data_folder = os.listdir(path+scenario)
        os.makedirs(path+scenario, exist_ok=True)
        for data_file in data_folder:
            data = pd.read_csv(path+scenario+'/'+data_file,sep=",")
            ts = data['Time'].max()
            for t in data['Time'] :
                
                if t % 1 ==0 : ## 5초 간격
                    idx_num = data.index[(data['Time']==t)]
                    if str(data['Health'].iloc[idx_num].iloc[0]) != 'Destroyed' :
                        if int(data['Location_y'].iloc[idx_num]) <= width-1 and int(data['Location_y'].iloc[idx_num]) >= 0 and int(data['Location_z'].iloc[idx_num]) <=height-1 and int(data['Location_z'].iloc[idx_num]) >= 0 :
                            if data['Force'].iloc[0] == 'Friendly': # Augmentation?
                                
                                if data['EntityType'].iloc[0] == 3:               #0번 채널 아군 human 1번 채널 아군 탈것 2번채널 적군 human  3번채널 적군 탈것
                                    coord_data[scenario_num][int(t)][0][int(data['Location_y'].iloc[idx_num])][int(data['Location_z'].iloc[idx_num])] +=1
                                else : 
                                    coord_data[scenario_num][int(t)][1][int(data['Location_y'].iloc[idx_num])][int(data['Location_z'].iloc[idx_num])] +=1
                            else :
                                if data['EntityType'].iloc[0] == 3:
                                    coord_data[scenario_num][int(t)][2][int(data['Location_y'].iloc[idx_num])][int(data['Location_z'].iloc[idx_num])] +=1
                                else:
                                    coord_data[scenario_num][int(t)][3][int(data['Location_y'].iloc[idx_num])][int(data['Location_z'].iloc[idx_num])] += 1
                    
        
        
        scenario_num+=1

        scenario_time.append(int(ts))
    return coord_data, scenario_time

def output_data_tagging(coord_data,scenario_time):
    health_fren = np.zeros((len(scenario_time),max_timestamp))
    health_oppo = np.zeros((len(scenario_time),max_timestamp))
    is_win = []
    
    for i in range(len(scenario_time)):
        
        for j in range(scenario_time[i]):
            health_fren[i][j] = np.sum(coord_data[i][j][0]) + np.sum(coord_data[i][j][1]) 
            health_oppo[i][j] =  np.sum(coord_data[i][j][2])
        if health_fren[i][scenario_time[i]-1] > health_oppo[i][scenario_time[i]-1]:
            is_win.append(1)
        elif health_fren[i][scenario_time[i]-1] < health_oppo[i][scenario_time[i]-1]:
            is_win.append(0)
        else :
            is_win.append(0.5)
        
    return health_fren, health_oppo, is_win

def damage_to_classification(damageRate):
    class_num = 5
    if damageRate<0:
        damageRate = 0
    damage_class = damageRate*class_num
    tmp = []
    for i in range(class_num):
        if i == int(damage_class) :
            tmp.append(1)
        elif i == class_num-1 and int(damage_class) >= class_num-1:
            tmp.append(1)
        else :
            tmp.append(0)
    return tmp


def data_sliding(coord_data, scenario_time,health_fren,health_oppo, is_win):
    
    total_batch = sum(scenario_time)
    data_x = np.zeros((total_batch,time,channels,width,height))
    data_y_wr = []
    data_y_fd = []
    data_y_od = []
    scenario_pointer = 0
    time_pointer = 0
    time_interval = 5 ## 5초 간격
    for i in range(total_batch):
        for t in range(time):
            if time_pointer-time+t >= 0:
                data_x[i][t] = coord_data[scenario_pointer][time_pointer-time+t].copy()
        wr = 0.5+ ((is_win[scenario_pointer]-0.5)*time_pointer/scenario_time[scenario_pointer])
        data_y_wr.append(wr)
        if health_fren[scenario_pointer][time_pointer] == 0 :
            tmp = damage_to_classification(0)
        else :
            if time_pointer+damage_time-1 <= scenario_time[scenario_pointer]:
                tmp = damage_to_classification((health_fren[scenario_pointer][time_pointer]-health_fren[scenario_pointer][time_pointer+damage_time-1])/health_fren[scenario_pointer][time_pointer])
            else :
                tmp = damage_to_classification((health_fren[scenario_pointer][time_pointer] -
                                                health_fren[scenario_pointer][scenario_time[scenario_pointer]]) /
                                               health_fren[scenario_pointer][time_pointer])
        data_y_fd.append(tmp)
        if health_oppo[scenario_pointer][time_pointer] == 0 :
            tmp = damage_to_classification(0)
        else :
            if time_pointer + damage_time - 1 <= scenario_time[scenario_pointer]:
                tmp = damage_to_classification((health_oppo[scenario_pointer][time_pointer]-health_oppo[scenario_pointer][time_pointer+damage_time-1])/health_oppo[scenario_pointer][time_pointer])
            else :
                tmp = damage_to_classification((health_oppo[scenario_pointer][time_pointer] -
                                                health_oppo[scenario_pointer][scenario_time[scenario_pointer]]) /
                                               health_oppo[scenario_pointer][time_pointer])
        data_y_od.append(tmp)
        
        time_pointer+=0 ## 5
        if int(scenario_time[scenario_pointer]) == time_pointer :
            time_pointer=0
            scenario_pointer+=1

    data_y = np.zeros((total_batch, 1+len(data_y_fd[0])+len(data_y_od[0]) ))
    for i in range(total_batch):
        data_y[i][0] = data_y_wr[i]
        for j in range(len(data_y_fd[0])) :
            data_y[i][j+1] = data_y_fd[i][j]
        for j in range(len(data_y_od[0])) :
            data_y[i][j+1+len(data_y_fd[0])] = data_y_od[i][j]



    return data_x, data_y

            



                                
def data_generating(path):
    folders = os.listdir(path)
    
    coordinateTransform(path,folders)
    coord_data,scenario_time = makeCoord(folders)
    health_fren, health_oppo, is_win = output_data_tagging(coord_data, scenario_time)
    data_x, data_y = data_sliding(coord_data, scenario_time,health_fren,health_oppo, is_win)
    return data_x, data_y
    """
    for scenario in folders:
        data_folder = os.listdir(path+scenario)
        for data_file in data_folder:
            data = pd.read_csv(path+scenario+'/'+data_file,sep=",")
    """
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Situation awareness Model')
    parser.add_argument('--weights', type=str, default="")
    parser.add_argument('--path', type=str, default="./VT-MAK_data/data_preprocessed/")
    

    args = parser.parse_args()
    # main(args)
    data_generating(args)
    
    

    

    
    