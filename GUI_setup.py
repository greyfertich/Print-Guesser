from tkinter import *
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot
from Paint import *
import scipy.misc
from Train import *
import threading

epoch = -1
root = Tk()
root.geometry("750x600")

def rightArrow(event):
    slide.set((slide.get() + 1))

def leftArrow(event):
    slide.set((slide.get() - 1))

def askQuestion():
    question.pack_forget()
    lab.pack(fill=X, side=TOP)
    slide.pack(side=BOTTOM, fill=X)
    root.bind('<Return>', getValue)
    root.bind('<Right>', rightArrow)
    root.bind('<Left>', leftArrow)

frame = Frame(root, width=1000, height=500, bd=0, bg='#2A353B')
title = Label(frame, text='Letter Reader', font=("Times", 64), fg='green', bg='#2A353B', bd=0)
question = Button(frame, text='Train', font=("Times", 16), highlightthickness=0, fg='green', bg='#394549', bd=0, activeforeground='#394549', activebackground='#2A353B', command=askQuestion)
lab = Label(text="Press Enter To Make Selection", font=('Times', 14), bg='#2A353B', bd=0, fg='dark gray')
slide = Scale(frame, orient=HORIZONTAL, font=("Times", 12), from_=1, to=50, fg='green', bd=0, bg='#2A353B', activebackground='#252F33', troughcolor='#394549', length=400, highlightthickness=0)
slide.set(26)
frame.pack()
root.configure(background="#2A353B")
root.title("Letter Reader")
title.pack(side=TOP, fill=X)
question.pack(side=BOTTOM)
ep, pe, t, eff, est = StringVar(), StringVar(), StringVar(), StringVar(), StringVar()
left_frame = Frame(frame, bg='#2A353B', bd=0)
title2 = Label(frame, font=('Times', 64), bg='#2A353B', bd=0, text='Training', fg='green')
esttime = Label(frame, font=('Times', 16), bg='#2A353B', fg='gray', bd=0, textvariable=est)
check = True;
net = neuralNetwork(1600, 256, 26, 0.01)

def training_thread(ep, pe, t, eff, est, epochs):
    global check
    tr = Train(net)
    tr.train(ep, pe, t, eff, est, epochs)
    check = False

def getValue(self):
    epoch = slide.get()
    title.pack_forget()
    question.pack_forget()
    slide.pack_forget()
    lab.pack_forget()
    title2.pack(side=TOP, fill=X)
    ep.set('Epoch = 1')
    epoch_counter = Label(left_frame, font=('Times', 24), bg='#2A353B', fg ='green', bd=0, textvariable=ep)
    epoch_counter.pack(side=TOP)
    pe.set('Performance = 0')
    perf = Label(left_frame, font=('Times', 24), bg='#2A353B', fg='green', bd=0, textvariable=pe)
    perf.pack()
    t.set('Time Per Epoch = 0')
    ti = Label(left_frame, font=('Times', 24), bg='#2A353B', fg='green', bd=0, textvariable=t)
    ti.pack()
    eff.set('Efficiency = 0')
    effic = Label(left_frame, font=('Times', 24), bg='#2A353B', fg='green', bd=0, textvariable=eff)
    effic.pack(side=BOTTOM)
    est.set('Time Remaining')
    esttime.pack(side=BOTTOM)
    # f = Figure(figsize=(3,2), dpi=100, facecolor='#2A353B', edgecolor='#00ff00')
    # a = f.add_subplot(111)
    # a.plot(range(8), [5,6,3,7,3,6,2,7])
    # graph = FigureCanvasTkAgg(f, frame)
    # graph.show()
    # graph.get_tk_widget().pack()
    left_frame.pack(side=LEFT, fill=X)
    train_thread = threading.Thread(target=training_thread, args=(ep, pe, t, eff, est, epoch))
    train_thread.start()

def continue_after():
    if not check:
        frame.pack_forget()
        left_frame.pack_forget()
        esttime.pack_forget()
        title2.pack_forget()
        cv = Paint(root, net)
    else:
        root.after(1000, continue_after)

root.after(1000, continue_after)
root.mainloop()
