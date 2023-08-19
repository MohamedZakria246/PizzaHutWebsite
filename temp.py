from tkinter import *
import tkinter as tk
from tkinter import ttk
import main
from main import *
root = tk.Tk()
root.title('Service Cancellation Predictor')
frame = Frame(root , width=1920 , height=1080)
frame.pack(expand=True, fill=BOTH)
canvas = Canvas(frame, width=1920 , height=1080 , scrollregion=(0,0,700,700))
scroll = Scrollbar(frame , orient= VERTICAL ,command=canvas.yview)
scroll.pack(side=RIGHT,fill=Y)
scroll.config(command=canvas.yview)
canvas.config(yscrollcommand=scroll.set)
canvas.config(width=1920,height=1080)
canvas.pack(expand=True,side=LEFT,fill=BOTH)
program_label = ttk.Label(frame, text="Methodology").place(x = 10 , y = 10 )

def check_algorithm_train():
    if(x1.get() == 1):
        logistic_tr = logistic_train()
        dash1 = ttk.Label(canvas, text=logistic_tr[0]).place(x = 250 , y = 190 )
        dash2 = ttk.Label(canvas, text=logistic_tr[1]).place(x=450, y=190)
        dash3 = ttk.Label(canvas, text=logistic_tr[2]).place(x=650, y=190)
        dash4 = ttk.Label(canvas, text=logistic_tr[3]).place(x=850, y=190)
    else:
        dash1 = ttk.Label(canvas, text="---------------------").place(x=250, y=190)
        dash2 = ttk.Label(canvas, text="---------------------").place(x=450, y=190)
        dash3 = ttk.Label(canvas, text="---------------------").place(x=650, y=190)
        dash4 = ttk.Label(canvas, text="---------------------").place(x=850, y=190)
    if(x2.get() == 1):
        svm_tr = svm_train()
        dash9 = ttk.Label(canvas, text=svm_tr[0]).place(x=250, y=310)
        dash10 = ttk.Label(canvas, text=svm_tr[1]).place(x=450, y=310)
        dash11 = ttk.Label(canvas, text=svm_tr[2]).place(x=650, y=310)
        dash12 = ttk.Label(canvas, text=svm_tr[2]).place(x=850, y=310)
    else:
        dash9 = ttk.Label(canvas, text="---------------------").place(x=250, y=310)
        dash10 = ttk.Label(canvas, text="---------------------").place(x=450, y=310)
        dash11 = ttk.Label(canvas, text="---------------------").place(x=650, y=310)
        dash12 = ttk.Label(canvas, text="---------------------").place(x=850, y=310)
    if(x3.get() == 1):
        id3_tr = Decision_train()
        dash17 = ttk.Label(canvas, text=id3_tr[0]).place(x=250, y=430)
        dash18 = ttk.Label(canvas, text=id3_tr[1]).place(x=450, y=430)
        dash19 = ttk.Label(canvas, text=id3_tr[2]).place(x=650, y=430)
        dash20 = ttk.Label(canvas, text=id3_tr[3]).place(x=850, y=430)
    else:
        dash17 = ttk.Label(canvas, text="---------------------").place(x=250, y=430)
        dash18 = ttk.Label(canvas, text="---------------------").place(x=450, y=430)
        dash19 = ttk.Label(canvas, text="---------------------").place(x=650, y=430)
        dash20 = ttk.Label(canvas, text="---------------------").place(x=850, y=430)
    if(x4.get() == 1):
        naive_tr = Naive_train()
        dash25 = ttk.Label(canvas, text=naive_tr[0]).place(x=250, y=550)
        dash26 = ttk.Label(canvas, text=naive_tr[1]).place(x=450, y=550)
        dash27 = ttk.Label(canvas, text=naive_tr[2]).place(x=650, y=550)
        dash28 = ttk.Label(canvas, text=naive_tr[3]).place(x=850, y=550)
    else:
        dash25 = ttk.Label(canvas, text="---------------------").place(x=250, y=550)
        dash26 = ttk.Label(canvas, text="---------------------").place(x=450, y=550)
        dash27 = ttk.Label(canvas, text="---------------------").place(x=650, y=550)
        dash28 = ttk.Label(canvas, text="---------------------").place(x=850, y=550)
    if(x5.get() == 1):
        random_tr = Random_Train()
        dash33 = ttk.Label(canvas, text=random_tr[0]).place(x=250, y=670)
        dash34 = ttk.Label(canvas, text=random_tr[1]).place(x=450, y=670)
        dash35 = ttk.Label(canvas, text=random_tr[2]).place(x=650, y=670)
        dash36 = ttk.Label(canvas, text=random_tr[3]).place(x=850, y=670)
    else:
        dash33 = ttk.Label(canvas, text="---------------------").place(x=250, y=670)
        dash34 = ttk.Label(canvas, text="---------------------").place(x=450, y=670)
        dash35 = ttk.Label(canvas, text="---------------------").place(x=650, y=670)
        dash36 = ttk.Label(canvas, text="---------------------").place(x=850, y=670)

def check_algorithm_test():
    if(x1.get() == 1):
        logistic_ts = logistic_test()
        dash5 = ttk.Label(canvas, text=logistic_ts[0]).place(x=250, y=220)
        dash6 = ttk.Label(canvas, text=logistic_ts[1]).place(x=450, y=220)
        dash7 = ttk.Label(canvas, text=logistic_ts[2]).place(x=650, y=220)
        dash8 = ttk.Label(canvas, text=logistic_ts[3]).place(x=850, y=220)
    else:
        dash5 = ttk.Label(canvas, text="---------------------").place(x=250, y=220)
        dash6 = ttk.Label(canvas, text="---------------------").place(x=450, y=220)
        dash7 = ttk.Label(canvas, text="---------------------").place(x=650, y=220)
        dash8 = ttk.Label(canvas, text="---------------------").place(x=850, y=220)
    if(x2.get() == 1):
        svm_ts = svm_test()
        dash13 = ttk.Label(canvas, text=svm_ts[0]).place(x=250, y=340)
        dash14 = ttk.Label(canvas, text=svm_ts[1]).place(x=450, y=340)
        dash15 = ttk.Label(canvas, text=svm_ts[2]).place(x=650, y=340)
        dash16 = ttk.Label(canvas, text=svm_ts[2]).place(x=850, y=340)
    else:
        dash13 = ttk.Label(canvas, text="---------------------").place(x=250, y=340)
        dash14 = ttk.Label(canvas, text="---------------------").place(x=450, y=340)
        dash15 = ttk.Label(canvas, text="---------------------").place(x=650, y=340)
        dash16 = ttk.Label(canvas, text="---------------------").place(x=850, y=340)
    if(x3.get() == 1):
        id3_ts = Decision_test()
        dash21 = ttk.Label(canvas, text=id3_ts[0]).place(x=250, y=460)
        dash22 = ttk.Label(canvas, text=id3_ts[1]).place(x=450, y=460)
        dash23 = ttk.Label(canvas, text=id3_ts[2]).place(x=650, y=460)
        dash24 = ttk.Label(canvas, text=id3_ts[3]).place(x=850, y=460)
    else:
        dash21 = ttk.Label(canvas, text="---------------------").place(x=250, y=460)
        dash22 = ttk.Label(canvas, text="---------------------").place(x=450, y=460)
        dash23 = ttk.Label(canvas, text="---------------------").place(x=650, y=460)
        dash24 = ttk.Label(canvas, text="---------------------").place(x=850, y=460)
    if(x4.get() == 1):
        naive_ts = Naive_test()
        dash29 = ttk.Label(canvas, text=naive_ts[0]).place(x=250, y=580)
        dash30 = ttk.Label(canvas, text=naive_ts[1]).place(x=450, y=580)
        dash31 = ttk.Label(canvas, text=naive_ts[2]).place(x=650, y=580)
        dash32 = ttk.Label(canvas, text=naive_ts[3]).place(x=850, y=580)
    else:
        dash29 = ttk.Label(canvas, text="---------------------").place(x=250, y=580)
        dash30 = ttk.Label(canvas, text="---------------------").place(x=450, y=580)
        dash31 = ttk.Label(canvas, text="---------------------").place(x=650, y=580)
        dash32 = ttk.Label(canvas, text="---------------------").place(x=850, y=580)
    if(x5.get() == 1):
        random_ts = Random_Test()
        dash37 = ttk.Label(canvas, text=random_ts[0]).place(x=250, y=700)
        dash38 = ttk.Label(canvas, text=random_ts[1]).place(x=450, y=700)
        dash39 = ttk.Label(canvas, text=random_ts[2]).place(x=650, y=700)
        dash40 = ttk.Label(canvas, text=random_ts[3]).place(x=850, y=700)
    else:
        dash37 = ttk.Label(canvas, text="---------------------").place(x=250, y=700)
        dash38 = ttk.Label(canvas, text="---------------------").place(x=450, y=700)
        dash39 = ttk.Label(canvas, text="---------------------").place(x=650, y=700)
        dash40 = ttk.Label(canvas, text="---------------------").place(x=850, y=700)

x1 = IntVar()
x2 = IntVar()
x3 = IntVar()
x4 = IntVar()
x5 = IntVar()
lb_1 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 40 )

logestic_cb = Checkbutton(canvas , text="Logestic Regression" , variable=x1 , onvalue=1, offvalue=0).place(x = 10 , y = 70 )
svm_cb = Checkbutton(canvas , text="SVM", variable=x2 , onvalue=1, offvalue=0).place(x = 160 , y = 70 )
id3_cb = Checkbutton(canvas , text="ID3", variable=x3  , onvalue=1, offvalue=0).place(x = 260 , y = 70 )
naive_cb = Checkbutton(canvas , text="Naive", variable=x4  , onvalue=1, offvalue=0).place(x = 360 , y = 70 )
random_cb = Checkbutton(canvas , text="Random", variable=x5  , onvalue=1, offvalue=0).place(x = 460 , y = 70 )

train_btn = Button(canvas, text="Train" ,width = 20 , height=1 , bg = 'silver' , command= check_algorithm_train).place(x = 10 , y = 100 )
test_btn = Button(canvas, text="Test", width = 20, height=1, bg = 'silver', command= check_algorithm_test).place(x = 180 , y = 100 )

lb_2 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 130 )

logestic_lb = ttk.Label(canvas, text="Logestic Regression").place(x = 10 , y = 160 )
accuracy_lb_1 = ttk.Label(canvas, text="Accuracy").place(x = 250 , y = 160 )
precision_lb_1 = ttk.Label(canvas, text="Precision").place(x = 450 , y = 160 )
recall_lb_1 = ttk.Label(canvas, text="Recall").place(x = 650 , y = 160 )
f1_lb_1 = ttk.Label(canvas, text="F1 Score").place(x = 850 , y = 160)
train_lb1 = ttk.Label(canvas, text="Train").place(x = 10 , y = 190 )


test_lb1 = ttk.Label(canvas, text="Test").place(x = 10 , y = 220 )

lb_3 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 250 )

svm_lb = ttk.Label(canvas, text="SVM").place(x = 10 , y = 280 )
accuracy_lb_2 = ttk.Label(canvas, text="Accuracy").place(x = 250 , y = 280 )
precision_lb_2 = ttk.Label(canvas, text="Precision").place(x = 450 , y = 280 )
recall_lb_2 = ttk.Label(canvas, text="Recall").place(x = 650 , y = 280 )
f1_lb_2 = ttk.Label(canvas, text="F1 Score").place(x = 850 , y = 280)
train_lb2 = ttk.Label(canvas, text="Train").place(x = 10 , y = 310 )

test_lb2 = ttk.Label(canvas, text="Test").place(x = 10 , y = 340 )

lb_4 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 370 )

id3_lb = ttk.Label(canvas, text="ID3").place(x = 10 , y = 400 )
accuracy_lb_3 = ttk.Label(canvas, text="Accuracy").place(x = 250 , y = 400 )
precision_lb_3 = ttk.Label(canvas, text="Precision").place(x = 450 , y = 400 )
recall_lb_3 = ttk.Label(canvas, text="Recall").place(x = 650 , y = 400 )
f1_lb_3 = ttk.Label(canvas, text="F1 Score").place(x = 850 , y = 400)
train_lb3 = ttk.Label(canvas, text="Train").place(x = 10 , y = 430 )

test_lb3 = ttk.Label(canvas, text="Test").place(x = 10 , y = 460 )

lb_5 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 490 )

naive_lb = ttk.Label(canvas, text="Naive").place(x = 10 , y = 520 )
accuracy_lb_4 = ttk.Label(canvas, text="Accuracy").place(x = 250 , y = 520 )
precision_lb_4 = ttk.Label(canvas, text="Precision").place(x = 450 , y = 520 )
recall_lb_4 = ttk.Label(canvas, text="Recall").place(x = 650 , y = 520 )
f1_lb_4 = ttk.Label(canvas, text="F1 Score").place(x = 850 , y = 520)
train_lb4 = ttk.Label(canvas, text="Train").place(x = 10 , y = 550 )

test_lb4 = ttk.Label(canvas, text="Test").place(x = 10 , y = 580 )

lb_6 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 610 )

random_lb = ttk.Label(canvas, text="Random").place(x = 10 , y = 640 )
accuracy_lb_5 = ttk.Label(canvas, text="Accuracy").place(x = 250 , y = 640 )
precision_lb_5 = ttk.Label(canvas, text="Precision").place(x = 450 , y = 640 )
recall_lb_5 = ttk.Label(canvas, text="Recall").place(x = 650 , y = 640 )
f1_lb_5 = ttk.Label(canvas, text="F1 Score").place(x = 850 , y = 640)
train_lb5 = ttk.Label(canvas, text="Train").place(x = 10 , y = 670 )

test_lb5 = ttk.Label(canvas, text="Test").place(x = 10 , y = 700 )

lb_7 = ttk.Label(canvas, text="--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------").place(x = 10 , y = 730 )

customer_data = ttk.Label(canvas, text="Customer Data" ).place(x = 10 , y = 760 )

id_lb = ttk.Label(canvas, text="Customer Id").place(x = 10 , y = 790 )
id_en = Entry(canvas , width = 30, relief = SUNKEN).place(x = 120 , y = 790 )
gender_lb = ttk.Label(canvas, text="Gender").place(x = 350 , y = 790 )
gender_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 790 )
senior_lb = ttk.Label(canvas, text="Senior Citizen").place(x = 690 , y = 790 )
senior_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 800 , y = 790 )

partner_lb = ttk.Label(canvas, text="Partener").place(x = 10 , y = 820 )
partner_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 120 , y = 820 )
dependert_lb = ttk.Label(canvas, text="Dependent").place(x = 350 , y = 820 )
dependert_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 820 )
tnure_lb = ttk.Label(canvas, text="Tenure").place(x = 690 , y = 820 )
tnure_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 800 , y = 820 )

phone_lb = ttk.Label(canvas, text="Phone Service").place(x = 10 , y = 850 )
phone_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 120 , y = 850 )
multipule_lb = ttk.Label(canvas, text="Miltipule Lines").place(x = 350 , y = 850 )
multipule_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 850 )
internet_lb = ttk.Label(canvas, text="Internet Service").place(x = 690 , y = 850 )
internet_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 800 , y = 850 )

security_lb = ttk.Label(canvas, text="Online Security").place(x = 10 , y = 880 )
security_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 120 , y = 880 )
backup_lb = ttk.Label(canvas, text="Online Backup").place(x = 350 , y = 880 )
backup_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 880 )
device_lb = ttk.Label(canvas, text="Device Protection").place(x = 690 , y = 880 )
device_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 800 , y = 880 )

tech_lb = ttk.Label(canvas, text="Tech Support").place(x = 10 , y = 910 )
tech_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 120 , y = 910 )
tv_lb = ttk.Label(canvas, text="Streaming Tv").place(x = 350 , y = 910 )
tv_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 910 )
movies_lb = ttk.Label(canvas, text="Streaming Movies").place(x = 690 , y = 910 )
movies_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 800 , y = 910 )

contract_lb = ttk.Label(canvas, text="Contract").place(x = 10 , y = 940 )
contract_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 120 , y = 940 )
paperles_lb = ttk.Label(canvas, text="Paperless Billing").place(x = 350 , y = 940 )
paperles_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 940 )
payment_lb = ttk.Label(canvas, text="Payment Method").place(x = 690 , y = 940 )
payment_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 800 , y = 940 )

month_lb = ttk.Label(canvas, text="Monthely Charges").place(x = 10 , y = 970 )
month_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 120 , y = 970 )
total_lb = ttk.Label(canvas, text="Total Charges").place(x = 350 , y = 970 )
total_en = Entry(canvas, width = 30, relief = SUNKEN).place(x = 460 , y = 970 )

pedict_btn = Button(canvas, text="Predict" ,width = 30, height=1, bg = 'silver' ).place(x = 10 , y = 1000 )
root.mainloop()