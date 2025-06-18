from tkinter import *
window = Tk()
label = Label(window, text="hello")
label.pack()

def process():
    print('button click')

button = Button(window, text ='click here!',
                bg = 'green', fg = 'blue', 
                width =60, height = 3, command= process)

button.pack()
window.mainloop()