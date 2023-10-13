import tkinter as tk


class Application:
    def __init__(self):
        self.root = tk.Tk()

        # Similar code as before but use self.root instead of self
        self.text_var = tk.StringVar()

        self.label = tk.Label(self.root, text="", textvariable=self.text_var, width=50, height=5)
        self.label.pack(pady=20)

        self.button = tk.Button(self.root, text="Open New Window", command=self.open_new_window)
        self.button.pack()

        self.root.mainloop()

    def open_new_window(self):
        self.top = tk.Toplevel(self.root)

        self.entry = tk.Entry(self.top)
        self.entry.pack(pady=20)

        self.submit_button = tk.Button(self.top, text="Submit", command=self.submit)
        self.submit_button.pack()

    def submit(self):
        self.text_var.set(self.entry.get())
        self.top.destroy()


app = Application()
