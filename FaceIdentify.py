import cv2
from Tkinter import *
import numpy as np
import os
import tkMessageBox
bin_n = 16

# chuyen anh xam
# ten
ten =["",]
# phat hien khuon mat
def detect_face(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('D:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2 ,minNeighbors= 5,minSize=(20,20));
    if(len(faces)==0):
        return None,None
    x,y,w,h = faces[0]
    return gray[y:y+w,x:x+h],faces[0]

# xu ly anh
def hog(img):
    gx = cv2.Sobel(img,cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img,cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))   
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     
    return hist
#tao du lieu training va labels
def create_data_traning(data_forder):
    dir = os.listdir(data_forder);
    # create two array to save data anh lables
    faces=[]
    labels=[]
    for dir_name in dir:
        if not dir_name.startswith('s'):
            continue
        label = int(dir_name.replace('s',""))
        subject_dir_path = data_forder + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path+"/"+image_name
            image= cv2.imread(image_path)
            face,rect = detect_face(image)
            if face is not None:
                fp = hog(face)
                faces.append(fp)
                labels.append(label)
    return faces,labels
# ve

def draw_rectangle(img, rect):
    (x,y,w,h) = rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,0),2)

def draw_text(img,text,x,y):
    cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
# tao du lieu training
svm = cv2.ml.SVM_create()
def train():
    faces ,labels = create_data_traning("data")
    dataTrain = np.float32(faces).reshape(-1,64)
    Labels = np.array(labels)[:,np.newaxis]
    # mo hinh svm
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(2.67)
    svm.setGamma(5.383)
    svm.train(dataTrain, cv2.ml.ROW_SAMPLE, Labels)
   # svm.save('svm_data.dat')
    tkMessageBox.showinfo("Thông Báo", "Máy đã học thành công")
#print type(labels)
def predict(img):
    face, rect = detect_face(img)
    if face is None:
        return
    else:
        hogdata2 = hog(face)
        imgtest = np.float32(hogdata2).reshape(-1,bin_n*4)
        result = svm.predict(imgtest)[1]
        label_text = ten[int(result[0])]
        draw_rectangle(img,rect)
        draw_text(img,label_text,rect[0],rect[1]-5)
        return img
def nhandien():
    cap =cv2.VideoCapture(0)
    while True:
            ret,frame = cap.read()
            pic = predict(frame)
            if pic is None:
                cv2.imshow('pic',frame)
            else:
                cv2.imshow('pic',pic)
        
            if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()
# ham tao du lieu
def hoc():
    face_cascade = cv2.CascadeClassifier('D:\\OpenCV\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    count = 0  
    name=stringTen.get()
    ten.append(name);
    i = len(ten)-1;
    path ='E:\\Ung dung nhan dien khuon mat\\data\\s%d'%i
    os.mkdir(path,755)
    cap = cv2.VideoCapture(0)
    while True:
        ret,img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5,minSize=(20,20))
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            if(count < 100):  
                cv2.imwrite('../Ung dung nhan dien khuon mat/data/s%d/anh%d.jpg'%(i,count),gray)
                count += 1
            else:
                cv2.putText(img,'tao du lieu xong roi ^^',(x,y),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow('img',img) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
              

    cap.release()
    cv2.destroyAllWindows()
# ham xoa thu muc
def xoafile():
    path ='E:\\Ung dung nhan dien khuon mat\\data'
    fileNeed=os.listdir(path)
    for item in fileNeed:
        lsimg=os.listdir("../Ung dung nhan dien khuon mat/data/%s"%item)
        for x in lsimg:
            os.remove("../Ung dung nhan dien khuon mat/data/%s/%s"%(item,x))
        os.removedirs("../Ung dung nhan dien khuon mat/data/%s"%item)
    os.mkdir(path,755)
    tkMessageBox.showinfo("Thông Báo", "xóa dữ liệu thành công")
# create GUi

def resetAction():
    stringTen.set("")
    
root = Tk()
stringTen = StringVar()
root.title("Nhận diên khuôn mặt")
root.resizable(width=True,height=True)
root.minsize(width=290,height=200)
Label(root,text="ỨNG DỤNG NHẬN DIÊN KHUÔN MẶT",fg="red",height=2,justify=CENTER).grid(row=0,columnspan=3)
Label(root,text="Tên của bạn :",fg="green",height=2).grid(row=1,column=0)
t=Entry(root,textvariable=stringTen,).grid(row=1,column=1)
t.pack(ipady=3)
root.geometry("400x400")
t = Text(r, height=20, width=40)
frameButton = Frame()
Button(frameButton,text="Tạo",fg="white",bg="violet",command = hoc).pack(side=LEFT)
Label(frameButton,text=" ").pack(side=LEFT)
Button(frameButton,text="Tiếp",fg="white",bg="violet",command=resetAction).pack(side=LEFT)
frameButton.grid(row=1,column=2)
Button(root,text="Cho Máy Học",fg="white",bg="Yellow",width=40,height=2,justify=CENTER,command=train).grid(row=3,columnspan=3)
Button(root,text="Nhận Diện",fg="white",bg="lightblue",width=40,height=2,justify=CENTER,command=nhandien).grid(row=4,columnspan=3)
Button(root,text="Clear Data",fg="white",bg="red",width=40,height=2,justify=CENTER,command=xoafile).grid(row=5,columnspan=3)
root.mainloop()

    

