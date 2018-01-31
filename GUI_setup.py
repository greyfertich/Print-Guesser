from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot
from Paint import *
import scipy.misc
from Train import *
import threading
from queue import Queue

plt.style.use('ggplot')
plt.rcParams['grid.color'] = '#005500'

epoch = -1
root = Tk() #Create window
root.geometry("750x600") #Set window size
q = Queue()

def rightArrow(event): #Increases slider value on right arrow press
    slide.set((slide.get() + 1))

def leftArrow(event): #Decreases slider value on left arrow press
    slide.set((slide.get() - 1))

def insertSlider(): #Displays slider after train button is pressed
    question.pack_forget()
    direction.pack(fill=X)
    lab.pack(fill=X, side=BOTTOM)
    slide.pack(side=BOTTOM, fill=X)
    root.bind('<Return>', lambda event: getValue(q))
    root.bind('<Right>', rightArrow)
    root.bind('<Left>', leftArrow)

frame = Frame(root, width=1000, height=500, bd=0, bg='#2A353B')
title = Label(frame, text='Print Guesser', font=("Times", 64), fg='green', bg='#2A353B', bd=0)
question = Button(frame, text='Train', font=("Times", 16), highlightthickness=0, fg='green', bg='#394549', bd=0, activeforeground='#394549', activebackground='#2A353B', command=insertSlider)
direction = Label(frame, text='Select the number of epochs', font=('Times', 14), bg='#2A353B', bd=0, fg='dark gray')
lab = Label(frame, text="Press Enter To Make Selection", font=('Times', 14), bg='#2A353B', bd=0, fg='dark gray')
slide = Scale(frame, orient=HORIZONTAL, font=("Times", 12), from_=1, to=50, fg='green', bd=0, bg='#2A353B', activebackground='#252F33', troughcolor='#394549', length=400, highlightthickness=0)
slide.set(26) #Sets initial slider value
frame.pack()
root.configure(background="#2A353B")
root.title("Print Guesser")
title.pack(side=TOP, fill=X)
question.pack(side=BOTTOM)
ep, pe, t, eff, est = StringVar(), StringVar(), StringVar(), StringVar(), StringVar()
left_frame = Frame(frame, bg='#2A353B', bd=0)
title2 = Label(frame, font=('Times', 64), bg='#2A353B', bd=0, text='Training', fg='green')
esttime = Label(frame, font=('Times', 16), bg='#2A353B', fg='gray', bd=0, textvariable=est)
check = True;
net = neuralNetwork(1600, 256, 26, 0.01)
performance_datalist = []
graph_frame = Frame(root)
graph_title = Label(graph_frame, text='Training Summary', fg='green', bg='#2A353B', bd=0, font=('Times', 64))
f = Figure(figsize=(5,4.25), dpi=100, facecolor='#2A353B', edgecolor='green')
a = f.add_subplot(111, facecolor='#2A353B')
can = FigureCanvasTkAgg(f, graph_frame)

def query():
    graph_frame.pack_forget()
    b.pack_forget()
    cv = Paint(root, net)

b = Button(root, text='Continue', font=("Times", 20), highlightthickness=0, fg='green', bg='#394549', bd=0, activeforeground='#394549', activebackground='#2A353B', command=query)

def training_thread(ep, pe, t, eff, est, epochs):
    global check, performance_datalist
    tr = Train(net)
    tr.train(ep, pe, t, eff, est, epochs)
    performance_datalist = tr.performance_data[:] #Copy list by value
    performance_datalist.insert(0,0.0)
    check = False

def getValue(q):
    global epoch
    epoch = slide.get()
    title.pack_forget()
    question.pack_forget()
    slide.pack_forget()
    lab.pack_forget()
    direction.pack_forget()
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
    est.set('')
    esttime.pack(side=BOTTOM)
    left_frame.pack(side=LEFT, fill=X)
    train_thread = threading.Thread(target=training_thread, args=(ep, pe, t, eff, est, epoch))
    train_thread.start()

def continue_after():
    global epoch, performance_datalist
    if not check:
        frame.pack_forget()
        left_frame.pack_forget()
        esttime.pack_forget()
        title2.pack_forget()
        graph_title.pack(side=TOP, fill=X)
        a.plot(range(epoch+1), performance_datalist, color='#00ff00')
        a.set_ylabel('Performace (%)')
        a.set_xlabel('Epoch')
        a.spines['left'].set_color('#00ff00')
        a.spines['bottom'].set_color('#00ff00')
        a.spines['right'].set_color('#2A353B')
        a.spines['top'].set_color('#2A353B')
        a.xaxis.label.set_color('#00ff00')
        a.yaxis.label.set_color('#00ff00')
        a.tick_params(axis='x', colors='#00ff00')
        a.tick_params(axis='y', colors='#00ff00')
        can.show()
        can._tkcanvas.pack(fill=X, expand=True)
        graph_frame.pack(side=TOP, fill=X, expand=True)
        b.pack(side=BOTTOM)
    else:
        root.after(1000, continue_after)

root.after(1000, continue_after)
root.mainloop()
check = True;
net = neuralNetwork(1600, 256, 26, 0.01)
performance_datalist = []

def training_thread(ep, pe, t, eff, est, epochs):
    global check, performance_datalist
    tr = Train(net)
    tr.train(ep, pe, t, eff, est, epochs)
    performance_datalist = tr.performance_data[:] #Copy list by value
    performance_datalist.insert(0,0.0)
    check = False

def getValue(q):
    global epoch
    epoch = slide.get()
    title.pack_forget()
    question.pack_forget()
    slide.pack_forget()
    lab.pack_forget()
    direction.pack_forget()
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
    est.set('')
    esttime.pack(side=BOTTOM)
    left_frame.pack(side=LEFT, fill=X)
    train_thread = threading.Thread(target=training_thread, args=(ep, pe, t, eff, est, epoch))
    train_thread.start()

def continue_after():
    global epoch, performance_datalist
    if not check:
        frame.pack_forget()
        left_frame.pack_forget()
        esttime.pack_forget()
        title2.pack_forget()
        plt.plot(range(epoch+1), performance_datalist)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Performance (% Correct)')
        plt.savefig('graph_image.png')
        im = Image.open('graph_image.png')
        im.save('graph_image.gif')
        cv = Paint(root, net)
    else:
        root.after(1000, continue_after)

root.after(1000, continue_after)
root.mainloop()
