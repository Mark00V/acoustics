# first draft setting of points
import tkinter as tk

#################################################
# tkinter static parameters
# window
WINDOW_HEIGHT = 800
WINDOWS_WIDTH = 1400
# canvas
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 600
DOTSIZE = 6 # click points in canvas
# Borderwidth (space between window frame and widgets and widgets between widgets, relative to windowsize)
BORDER_WIDTH = 0.01
#################################################

class FEMGUI:


    def __init__(self):
        self.polygon_coordinates = [[0, 0]]
        self.firstclick_canvas = True



    def on_canvas_click(self, event):
        # Get coordinates of right click

        x, y = event.x, event.y
        self.polygon_coordinates.append([x, y])
        self.polygon_coordinates_tmp += f"[{x},{y}]  "
        self.polygon_coordinates_str.set(self.polygon_coordinates_tmp)


        # create lines between points
        if self.firstclick_canvas == True:
            self.firstclick_canvas = False
            self.canvas.create_line(0, CANVAS_HEIGHT, x, y, fill="black", width=2)

        if 'prevpoint' in FEMGUI.on_canvas_click.__dict__ and not self.firstclick_canvas:
            self.canvas.create_line(FEMGUI.on_canvas_click.prevpoint[0], FEMGUI.on_canvas_click.prevpoint[1], x, y, fill="black", width=3)
        FEMGUI.on_canvas_click.prevpoint = (x,y) # TODO: Why does this not work with self????

        # Clear content
        self.coord_var_x.set('0')
        self.coord_var_y.set('0')

        # Display coordinates
        self.coord_var_x.set(f"{x}")
        self.coord_var_y.set(f"{y}")

        self.canvas.create_oval(x - DOTSIZE, y - DOTSIZE, x + DOTSIZE, y + DOTSIZE, outline="black", fill="white")

    def start(self):


        # Start tkinter
        root = tk.Tk()
        root.title("Main")
        root.geometry(f"{WINDOWS_WIDTH}x{WINDOW_HEIGHT}")

        # Canvas widget for setting nodes of polygon
        self.canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="gray")
        self.canvas.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH, rely=BORDER_WIDTH)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        # set initial node
        self.canvas.create_oval(0 - DOTSIZE, CANVAS_HEIGHT - DOTSIZE, 0 + DOTSIZE, CANVAS_HEIGHT + DOTSIZE, outline="black", fill="white")

        # Inital values for widgets
        self.coord_var_x = tk.StringVar()
        self.coord_var_y = tk.StringVar()
        self.polygon_coordinates_str = tk.StringVar()
        self.coord_var_x.set('0')
        self.coord_var_y.set('0')
        self.polygon_coordinates_str.set(str(self.polygon_coordinates[0]))
        self.polygon_coordinates_tmp = str(self.polygon_coordinates[0]) + ' '

        # Dynamic display coordinates last click
        coord_entry_width = 6
        coord_label = tk.Label(root, text="Coordinates:", font=("Arial", 12))
        coord_label.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))
        coord_label_x = tk.Label(root, text="X:", font=("Arial", 12))
        coord_label_x.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.075, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))
        coord_label_y = tk.Label(root, text="Y:", font=("Arial", 12))
        coord_label_y.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.15, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))

        coord_entry_x = tk.Entry(root, textvariable=self.coord_var_x, font=("Arial", 12), width=coord_entry_width, justify='left')
        coord_entry_x.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.095, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))
        coord_entry_y = tk.Entry(root, textvariable=self.coord_var_y, font=("Arial", 12), width=coord_entry_width, justify='left')
        coord_entry_y.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.168, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT))

        # Display list of coordinates
        coord_set_label = tk.Label(root, text="All polygon vertices:", font=("Arial", 12))
        coord_set_label.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT)+0.05)
        polygon_coordinates = tk.Entry(root, textvariable=self.polygon_coordinates_str, state='readonly', font=("Arial", 12), width=100, justify='left')
        polygon_coordinates.place(relx=1-(CANVAS_WIDTH/WINDOWS_WIDTH)-BORDER_WIDTH+0.115, rely=2*BORDER_WIDTH + (CANVAS_HEIGHT/WINDOW_HEIGHT)+0.05)


        # Run the Tkinter main loop
        root.mainloop()


def main():
    start = FEMGUI()
    start.start()

if __name__ == "__main__":
    main()