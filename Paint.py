from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import csv
from random import randint
from Neural_Net import *
import time

class Paint:
    def __init__(self, master, net):
        self.master = master
        self.inodes = net.inodes
        self.hnodes = net.hnodes
        self.onodes = net.onodes
        self.lr = net.lr
        self.net = net #Creates Neural net
        self.ch = IntVar() #Displays training letter
        self.ch.set(randint(65,90))
        self.pred_text = StringVar()
        self.pred_text.set("Prediction")
        self.draw_count = 0
        self.width = 500
        self.height = 360
        self.max_height = 0
        self.min_height = self.height
        self.max_width = 0
        self.min_width = self.width
        self.line_count = 0
        self.crop_size = 40
        self.check_drawn = False
        self.saved_image = Image.new("L", (self.width, self.height), (255)) #Creates image for saving drawing
        self.saved_image_draw = ImageDraw.Draw(self.saved_image)
        self.neural_height = 500
        self.neural_width = 250
        self.neural_image = Image.new("RGBA", (self.neural_width, self.neural_height), (255,255,255)) #Image of neural net
        self.neural_image_draw = ImageDraw.Draw(self.neural_image)
        self.draw_color = '#00ff00'
        self.title = Label(master, text='Query Neural Net', bg='#2A353B', font=('Times', 64), fg='green')
        self.title.pack()
        self.newFrame = Frame(master, bg='#2A353B')
        self.directions = Label(self.newFrame, bg='#2A353B', fg='gray', font=('Times', 16), text='Draw a letter in the field below\nPress enter when finished')
        self.directions.pack(side=TOP)
        self.draw = Canvas(self.newFrame, bg='#2A353B', width=self.width, height=self.height, bd=0, highlightthickness=0) #Creates canvas for drawing
        self.draw.bind('<Button-1>', self.getCoordinates) #Left-click starts line
        self.draw.bind('<B1-Motion>', self.draw_lines) #Motion continues line
        self.draw.bind('<Button-3>', self.clear_canvas) #Right click clears canvas
        self.draw.pack(side=TOP)
        self.newFrame.pack(side=LEFT)
        green = 255
        self.text_color = '#%02x%02x%02x' % (255-green, green, 0)
        self.prediction = Label(self.newFrame, font=("Times", 64), textvariable=self.pred_text, fg='#00ff00', bg='#2A353B', bd=0)
        self.prediction.pack(side=BOTTOM)
        img = PhotoImage(file='neural_image_start.gif')
        self.net_visual = Label(master, image=img, bg='#2A353B', bd=0)
        self.net_visual.image = img
        self.net_visual.pack(side=RIGHT)

    def getCoordinates(self, event):

        self.line_count += 1

        self.x = event.x
        self.y = event.y
        #Checks for new width min/max
        if self.line_count == 1:
            if self.x > self.max_width:
                self.max_width = self.x
            if self.x < self.min_width:
                self.min_width = self.x

            #Checks for new height min/max
            if self.y > self.max_height:
                self.max_height = self.y
            if self.y > self.min_height:
                self.min_height = self.y

    def draw_lines(self, event):
        self.newX = event.x
        #Checks for new Width min/max
        if self.newX > self.max_width:
            self.max_width = self.newX
        if self.newX < self.min_width:
            self.min_width = self.newX

        self.newY = event.y
        #Checks for new height min/max
        if self.newY > self.max_height:
            self.max_height = self.newY
        if self.newY < self.min_height:
            self.min_height = self.newY

        self.draw.create_line(self.x, self.y, self.newX, self.newY, fill=self.text_color,
                              smooth=True, width=10)
        self.saved_image_draw.line((self.x, self.y, self.newX, self.newY), fill='black', width=10)
        self.x = self.newX
        self.y = self.newY

    def clear_canvas(self, event):

        self.draw_count += 1

        #Get height and width range
        if self.min_width < 0: self.min_width = 0
        if self.min_height < 0: self.min_height = 0
        if self.max_width > self.width: self.max_width = self.width
        if self.max_height > self.height: self.max_height = self.height

        self.height_range = self.max_height - self.min_height
        self.width_range = self.max_width - self.min_width

        if self.height_range > self.width_range:
            self.avg = self.height_range / 2
            self.avg_w = (self.min_width + self.max_width) / 2
            self.min_width = self.avg_w - self.avg
            self.max_width = self.avg_w + self.avg
        else:
            self.avg = self.width_range / 2
            self.avg_h = (self.min_height + self.max_height) / 2
            self.min_height = self.avg_h - self.avg
            self.max_height = self.avg_h + self.avg

        self.saved_image3 = self.saved_image.crop((self.min_width, self.min_height, self.max_width, self.max_height))
        self.saved_image2 = self.saved_image3.resize((self.crop_size,self.crop_size), Image.ANTIALIAS)
        self.saved_image3.save("test_drawing3.png")
        self.saved_image2.save("test_drawing.png")
        self.saved_image.save("test_drawing2.png")
        self.saved_image_draw.rectangle((0, 0, self.width, self.height), fill='white')
        self.draw.delete("all")
        self.neural_image_draw.rectangle((0,0,250,500), fill='white')

        #reset the max/him height and width
        self.max_height = 0
        self.min_height = self.height
        self.max_width = 0
        self.min_width = self.width

        self.final_image = Image.open("test_drawing.png")

        self.final_image.load()
        self.data = np.asarray(self.final_image)
        self.datalist = self.data.flatten().tolist()
        self.datalist.insert(0, (self.ch.get() - 65))
        new_input = ((numpy.asfarray(self.datalist[1:]) / 255.0 * 0.99) + 0.01) #Gets array of single input data
        final = self.net.query(new_input).flatten().tolist() #Converts array to list
        st = ('{} ({}%)'.format(chr((np.argmax(final))+65), int(final[np.argmax(final)]*100)))
        self.pred_text.set(st)
        green = int(final[np.argmax(final)] * 255)
        self.text_color = '#%02x%02x%02x' % (255-green, green, 0)
        self.prediction.configure(fg=self.text_color)
        self.title.configure(fg=self.text_color)

        self.inodes_display = int(self.inodes ** 0.5)
        self.hnodes_display = int(self.hnodes ** 0.5)
        self.onodes_display = self.onodes

        input_separation  = round((self.neural_height / (self.inodes_display + 1.0)), 2)
        hidden_separation = self.neural_height / (self.hnodes_display + 1.0)
        output_separation = self.neural_height / (self.onodes_display + 1.0)
        inode_size = input_separation / 4.0
        hnode_size = hidden_separation / 4.0
        onode_size = output_separation / 4.0
        self.neural_image_draw.rectangle((0,0,self.neural_width, self.neural_height), fill='#ffffff')
        #Draw neural net
        for counter in range(self.inodes_display): #Draws neural values to image
            for counter2 in range(self.hnodes_display):
                num = numpy.random.randint(10)
                num2 = numpy.random.randint(100) + 155
                color = self.text_color
                r_color = '#%02x%02x%02x' % (num2, 0, 0)
                if num > 7: self.neural_image_draw.line((25,input_separation + (counter*input_separation),125,hidden_separation + (counter2*hidden_separation)),fill=color)
                if num < 2: self.neural_image_draw.line((25,input_separation + (counter*input_separation),125,hidden_separation + (counter2*hidden_separation)),fill=r_color)
        for counter in range(self.hnodes_display):
            for counter2 in range(self.onodes_display):
                num = numpy.random.randint(5)
                num2 = numpy.random.randint(100) + 155
                color = self.text_color
                if num == 1:
                    num3 = numpy.random.randint(200)
                    color2 = '#%02x%02x%02x' % (num3, 0, 0)
                    self.neural_image_draw.line((125,hidden_separation + (counter*hidden_separation),225,output_separation + (counter2*output_separation)),fill=color2)
                if counter2 == np.argmax(final):
                    self.neural_image_draw.line((125,hidden_separation + (counter*hidden_separation),225,output_separation + (counter2*output_separation)),fill=color)

        for counter in range(self.inodes_display): self.neural_image_draw.ellipse((25 - inode_size,(input_separation - inode_size) + (counter*input_separation),25 + inode_size,(input_separation + inode_size) + (counter*input_separation)),fill="black")
        for counter in range(self.hnodes_display): self.neural_image_draw.ellipse((125 - inode_size,(hidden_separation - inode_size) + (counter*hidden_separation),125 + inode_size,(hidden_separation + inode_size) + (counter*hidden_separation)),fill="black")
        for counter in range(self.onodes_display): self.neural_image_draw.ellipse((225 - inode_size,(output_separation - inode_size) + (counter*output_separation),225 + inode_size,(output_separation + inode_size) + (counter*output_separation)),fill="black")

        self.neural_image.save("neural_image.gif", 'GIF', transparency=0)
        self.net_visual.pack_forget()
        img = PhotoImage(file='neural_image.gif')
        self.net_visual = Label(self.master, image=img, bg='#2A353B', bd=0)
        self.net_visual.image = img
        self.net_visual.pack(side=RIGHT)
