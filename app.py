import argparse
import logging
import tkinter as tk
from tkinter import *

import tkinter.font as font
import webbrowser
import random

from utils.captureImages import CaptureImages 
from utils.train import Train

class RegistrationModule:
    def __init__(self,logFileName):

        self.logFileName = logFileName
        self.window = tk.Tk()
        self.window.title('Face Recognition and tracking')

        self.window.resizable(0,0)
        window_height = 600
        window_width = 900

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x_coordinate = int((screen_width / 2) - (window_width/2))
        y_coordinate = int((screen_height / 2) - (window_height/2))

        self.window.geometry("{}x{}+{}+{}".format(window_width,window_height,x_coordinate,y_coordinate))
        self.window.configure(background='#ffffff')

        self.window.grid_rowconfigure(0,weight=1)
        self.window.grid_columnconfigure(0,weight=1)

        header = tk.Label(self.window,text='Employee Monitoring System',width=80, height=2, fg="white", bg="#363e75",
                font=('times', 18, 'bold', 'underline'))

        header.place(x=0,y=0)
        clientId = tk.Label(self.window,text='Client Id',width=10, height=2, fg="white", bg="#363e75", font=('times', 15))
        clientId.place(x=80,y=80)

        displayVariable = StringVar()
        self.clientIDTxt = tk.Entry(self.window,width=20, text=displayVariable, bg="white", fg="black",
                            font=('times', 15, 'bold'))
        self.clientIDTxt.place(x=205,y=80)

        empID = tk.Label(self.window, text="EmpID", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empID.place(x=450, y=80)

        self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empIDTxt.place(x=575, y=80)

        empName = tk.Label(self.window, text="Emp Name", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empName.place(x=80, y=140)

        self.empNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empNameTxt.place(x=205, y=140)

        emailId = tk.Label(self.window, text="Email ID ", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        emailId.place(x=450, y=140)

        self.emailIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.emailIDTxt.place(x=575, y=140)

        mobileNo = tk.Label(self.window, text="Mobile No", width=10, fg="white", bg="#363e75", height=2,
                            font=('times', 15))
        mobileNo.place(x=450, y=140)

        self.mobileNoTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.mobileNoTxt.place(x=575, y=140)

        lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
                        font=('times', 15))
        self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
                                activebackground="#e47911",font=('times',15))

        self.message.place(x=220, y=220)
        lbl3.place(x=80, y=260)

        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
                           font=('times', 15))
        self.message.place(x=205, y=260)


        takeImg = tk.Button(self.window, text="Take Images", fg="white", command=self.collectImagesFromCamera, bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '))
        takeImg.place(x=80, y=350)

        trainImg = tk.Button(self.window, text="Train Images",command=self.trainModel, fg="white", bg="#363e75", width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        trainImg.place(x=350, y=350)

        predictImg = tk.Button(self.window, text="Predict", fg="white", bg="#363e75",
                             width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        predictImg.place(x=600, y=350)

        quitWindow = tk.Button(self.window, text="Quit", command=self.closeWindow, fg="white", bg="#363e75", width=10, height=2,
                               activebackground="#118ce1", font=('times', 15, 'bold'))
        quitWindow.place(x=650, y=510)

        label = tk.Label(self.window)

        self.window.mainloop()

    def trainModel(self):
        trainModel = Train()
        trainModel.train()

        notification = 'your model is ready now'
        self.message.configure(text=notification)

    def collectImagesFromCamera(self):
        clientIdVal = (self.clientIDTxt.get())
        empIDVal = (self.empIDTxt.get())
        name = (self.empNameTxt.get())
        ap = argparse.ArgumentParser()

        ap.add_argument("--faces", default=30,help='number of faces that camera will get')
        ap.add_argument("--output",default="./datasets/train/"+str(clientIdVal) + '_' + str(empIDVal) + '_' + str(name),
                                help='path to faces output')

        args = vars(ap.parse_args())

        trngDataCollector = CaptureImages(args)
        trngDataCollector.capture()

        notification = "We have collected " + str(args['faces']) + " images for training."
        self.message.configure(text=notification)


    def closeWindow(self):
        self.window.destroy()


if __name__ == '__main__':
    register = RegistrationModule('proceduralLog.txt')
