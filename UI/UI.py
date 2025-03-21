import tkinter as tk

def say_hi():
    print("Hi")
    
def say_bye():
    print("Bye")


root = tk.Tk()
    
root.title("My GUI")

btn = tk.Button(root, text="Say Hi", command=say_hi)
btn.pack(pady=20)
btn = tk.Button(root, text="Say Bye", command=say_bye)
btn.pack(pady=20)


root.mainloop()
