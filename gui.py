#!/usr/bin/python3
from tkinter import * # note that module name has changed from Tkinter in Python 2 to tkinter in Python 3
panel1 = PanedWindow()
panel1.pack(fill = BOTH, expand =1)
left = Canvas(panel1)
panel1.add(left)
panel2 = PanedWindow( panel1, orient = VERTICAL)
panel1.add(panel2)
first = Button(panel2, text ="Reduce X")
panel2.add(first)
second = Button(panel2, text ="Reduce y")
panel2.add(second)
third = Button(panel2, text ="Info")
panel2.add(third)
mainloop()
