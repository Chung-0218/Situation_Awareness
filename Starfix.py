from time import sleep
import tkinter.ttk as ttk
from tkinter import *
import pandas as pd

# 예측 결과 불러오기
scenario = pd.read_csv("predict.csv", encoding='utf8')
time_stamp = scenario.iloc[:, 1]
win_rate = scenario.iloc[:, 2]
fren_output = scenario.iloc[:, 3:8]
oppo_output = scenario.iloc[:, 8:13]

root = Tk()
root.title("Battlefield Awareness Module 1.0")
root.geometry("510x180")

for t in range(len(time_stamp)):
    # [1] Top label 추출 (컬럼명 기준)
    top1acc = '승률'  # 승률은 하나라서 고정 표현 가능
    top2acc = fren_output.columns[fren_output.iloc[t].argmax()]
    top3acc = oppo_output.columns[oppo_output.iloc[t].argmax()]

    # [2] 해당 시점의 값 추출
    top1per = win_rate.iloc[t]
    top2per = fren_output.iloc[t].max()
    top3per = oppo_output.iloc[t].max()

    # [3] 라벨 텍스트 UI
    label1 = Label(root, text="승률", width=15, bg='white', height=2, relief="solid")
    label1.place(x=30, y=30)

    label2 = Label(root, text="아군 피해도", width=15, bg='white', height=2, relief="solid")
    label2.place(x=30, y=70)

    label3 = Label(root, text="적군 피해도", width=15, bg='white', height=2, relief="solid")
    label3.place(x=30, y=110)

    label11 = Label(root, text=top1acc, width=15, height=2, bg='red', relief="groove")
    label11.place(x=150, y=30)

    label21 = Label(root, text=top2acc, width=15, height=2, bg='lightgreen', relief="groove")
    label21.place(x=150, y=70)

    label31 = Label(root, text=top3acc, width=15, height=2, bg='lightgreen', relief="groove")
    label31.place(x=150, y=110)

    label12 = Label(root, text=f"{top1per:.2f}", width=15, height=2, bg='white', fg='red', relief="groove")
    label12.place(x=270, y=30)

    label22 = Label(root, text=f"{top2per:.2f}", width=15, height=2, bg='white', fg='red', relief="groove")
    label22.place(x=270, y=70)

    label32 = Label(root, text=f"{top3per:.2f}", width=15, height=2, bg='white', fg='red', relief="groove")
    label32.place(x=270, y=110)

    pbar1 = ttk.Progressbar(root, value=top1per, maximum=1, mode="determinate")
    pbar1.place(x=390, y=35)

    pbar2 = ttk.Progressbar(root, value=top2per, maximum=1, mode="determinate")
    pbar2.place(x=390, y=75)

    pbar3 = ttk.Progressbar(root, value=top3per, maximum=1, mode="determinate")
    pbar3.place(x=390, y=115)

    sleep(1)
    root.update()

root.mainloop()
