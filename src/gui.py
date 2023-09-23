import tkinter as tk
from tkinter import messagebox as mb, filedialog as fd
from os import getcwd
from os.path import join
from svhn_detection import SVHN_Detection
from webbrowser import open as web_open

import sys
class Console(tk.Text):
    """
    From [rdbende's answer on Reddit](https://www.reddit.com/r/Tkinter/comments/nmx0ir/how_to_show_terminal_output_in_gui/)
    \nThis class changes the `stdout` to display it on a tkinter Text widget.
    Modified this class to have a `flush` method.
    """
    def __init__(self, *args, **kwargs):
        kwargs.update({"state": "disabled"})
        tk.Text.__init__(self, *args, **kwargs)
        self.bind("<Destroy>", self.reset)
        self.old_stdout = sys.stdout
        sys.stdout = self
    
    def delete(self, *args, **kwargs):
        self.config(state="normal")
        self.delete(*args, **kwargs)
        self.config(state="disabled")
    
    def write(self, content):
        self.config(state="normal")
        self.insert("end", content)
        self.see("end")
        self.config(state="disabled")

    def flush(self):
        self.config(state="normal")
        self.update()
        self.config(state="disabled")
    
    def reset(self, event):
        sys.stdout = self.old_stdout

class SVHN_GUI():
    """
    GUI to train a `tensorflow.keras` model for the SVHN dataset.
    """
    version = "1.0"
    author = "Lucien Schneider"
    github_repo = "https://github.com/3DTS/ML_Project"
    readme = "https://github.com/3DTS/ML_Project/blob/main/README.md"
    def __init__(self, window):
        """
        ### Parameters
        1. window : tk.Tk() 
                - Instance of tkinter object.
        """
        self.window = window
        self.train_directory = getcwd()
        self.train_data = join(self.train_directory, "Detection - SVHN", "train_32x32.mat")
        self.test_data = join(self.train_directory, "Detection - SVHN", "test_32x32.mat")

        self.svhn = SVHN_Detection()

        self.filetypes = (
        ("MATLAB file", "*.mat"),
        ("All files", "*.*")
        )

        self.__setupWidgets()
        
    def __setupWidgets(self):
        """Place tkinter widgets on root object.
        Calls `self.__setupMenubar`.
        """
        self.__setupMenubar()

        # GUI 
        self.window.title("SVHN Detection")
        self.window.protocol("WM_DELETE_WINDOW", self.__askSave)

        # 1. Frame
        self.frm_load_ds = tk.Frame(self.window, relief="groove", borderwidth=5)
        self.frm_load_ds.grid(row=0, column=0, columnspan=1)

            # Label (title)
        self.lbl_load_ds = tk.Label(self.frm_load_ds, 
                                    text="Load Dataset")
        self.lbl_load_ds.grid(row=0, column=0, sticky="nsew", columnspan=2)

            # Button (Choose directory)
        self.btn_dir = tk.Button(self.frm_load_ds, 
                                 text="Choose files", 
                                 command=self.__setFiles)
        self.btn_dir.grid(row=1,column=0, sticky="w")
            
            # Button (Start training)
        self.btn_start_learn = tk.Button(self.frm_load_ds, 
                                         text="Start training", 
                                         command=self.__trainMLModel)
        self.btn_start_learn.grid(row=1, column=1, sticky="e")

            # Label (Num of Epochs)
        self.lbl_epochs = tk.Label(self.frm_load_ds,
                                   text="Number of Epochs:")
        self.lbl_epochs.grid(row=2, column=0, sticky="ew")

            # Spinbox (Set num of epochs)
        self.sb_epochs = tk.Spinbox(self.frm_load_ds, 
                                    from_=1., to=50., increment=1,
                                    command=self.__setEpochs,
                                    width=10)
        self.sb_epochs.grid(row=2, column=1, sticky="ew")


        # 2. Frame (Load and save model)
        self.frm_load_model = tk.Frame(self.window, relief="groove", borderwidth=5)
        self.frm_load_model.grid(row=1, column=0, columnspan=1, sticky="ew")

            # Label (title)
        self.lbl_load_model = tk.Label(self.frm_load_model, 
                                       text="Load and save model")
        self.lbl_load_model.grid(row=0, column=0, sticky="nsew", columnspan=2)

            # Buttons
        self.btn_ld_model = tk.Button(self.frm_load_model, 
                                       text="Load model", 
                                       command=self.__loadFromDir)
        self.btn_ld_model.grid(row=1, column=0, sticky="w")

        self.btn_save_model = tk.Button(self.frm_load_model, 
                                       text="Save model", 
                                       command=self.__saveToDir)
        self.btn_save_model.grid(row=1, column=1, sticky="e")
        

        # 3. Frame (validation data)
        self.frm_validate = tk.Frame(self.window, relief="groove", borderwidth=5)
        self.frm_validate.grid(row=2, column=0, columnspan=1, sticky="ew")
        
            # Label (title)
        self.lbl_validate = tk.Label(self.frm_validate, 
                                     text="Validate model")
        self.lbl_validate.grid(row=0, column=0, sticky="ew", columnspan=2)

            # Button (set validation data)
        self.btn_val_data = tk.Button(self.frm_validate,
                                     text="Evaluate",
                                     command=self.__evaluateModel)
        self.btn_val_data.grid(row=1, column=0)
            
            # Buttons (show images)
        self.btn_image = tk.Button(self.frm_validate,
                                   text="Images",
                                   command=self.__showImageGrid)
        self.btn_image.grid(row=1, column=1)

        self.btn_graph = tk.Button(self.frm_validate,
                                   text="Graph",
                                   command = self.__showLAGraph)
        self.btn_graph.grid(row=1, column=2)

        
        # Console (output)
        self.s = tk.Scrollbar(self.window, orient="vertical")
        self.s.grid(row=0, column=2, rowspan=3, sticky="ns")

        self.console = Console(self.window, background="#000000", foreground="#ffffff", width=100)
        self.console.grid(row=0, column=1, rowspan=3)

        self.s.config(command=self.console.yview)
        self.console.config(yscrollcommand=self.s.set)

        self.window.geometry()
        self.window.update()
        self.window.minsize(self.window.winfo_width(),self.window.winfo_height())
        self.window.resizable(False,False)   

    def __setupMenubar(self):
        """Create a menubar on the top of the window with two separate menus "Files" and "Help".
        - "Files" has all options displayed as buttons.
        - "Help" contains information and the documentation for the code.
        
        """

        self.menubar = tk.Menu(self.window)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="New model", command=self.__setFiles)
        self.filemenu.add_command(label="Open model", command=self.__loadFromDir)
        self.filemenu.add_command(label="Save model", command=self.__saveToDir)
        self.filemenu.add_command(label="Train data", command=self.__setFiles)
        self.filemenu.add_command(label="Validation data", command=self.__setFiles)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.__askSave)
        self.menubar.add_cascade(label="File", menu=self.filemenu)

        self.helpmenu = tk.Menu(self.menubar, tearoff=0)
        self.helpmenu.add_command(label="Help", command=self.__mbOpenHelp)
        self.helpmenu.add_command(label="About...", command=self.__menubarAbout)
        self.helpmenu.add_separator()
        self.helpmenu.add_command(label="Visit GitHub repo", command=self.__mbVisitGitHub)
        self.menubar.add_cascade(label="Help", menu=self.helpmenu)

        self.window.config(menu=self.menubar)

    def __trainMLModel(self):
        """Method to set up and call method for model training.\n
        Reads the state of spinbox object and transfers this value to the `svhn_dataset` object.
        Deactivates all buttons and activates them, when training is finished to avoid user interaction.
        All `tensorflow`-messages will be displayed on `Console`-widget.
        ### Catches
        - `RunRuntimeError` - A Messagebox displays the errormessage.
        - `AttributeError` - A Messagebox displays the errormessage.
        - `FileNotFoundError` - A Messagebox displays a possible solution.
        """
        e = int(self.sb_epochs.get())
        self.svhn.setNumEpochs(e)

        self.btn_dir.config(state="disabled")
        self.btn_val_data.config(state="disabled")
        self.btn_start_learn.config(state="disabled")
        self.btn_graph.config(state="disabled")
        self.btn_image.config(state="disabled")
        self.btn_save_model.config(state="disabled")
        self.btn_ld_model.config(state="disabled")
        try:
            self.svhn.createAndTrain()
        except RuntimeError as e:
            mb.showwarning(title="Runtime error",
                           message="\n{}".format(e))
        except AttributeError as e:
            mb.showwarning(title="Attribute not found",
                           message="No attributes found for training. \nPlease load training dataset!\n{}".format(e))
        except FileNotFoundError:
            mb.showwarning(title="File not found", 
                           message="The specified file does not exist.")
        finally:
            self.btn_dir.config(state="normal")
            self.btn_val_data.config(state="normal")
            self.btn_start_learn.config(state="normal")
            self.btn_graph.config(state="normal")
            self.btn_image.config(state="normal")
            self.btn_save_model.config(state="normal")
            self.btn_ld_model.config(state="normal")
         
    def __evaluateModel(self):
        """Method for button command.
        Calls `evaluateLoss` method from class `SVHN_Detection` and prints loss and accuracy to `Console` widget.
        """
        self.svhn.evaluateLoss(print_to_console=True)

    def __setFiles(self, title="Open Files"):
        """Asks user to specify train and validation datasets of type ".mat" 
        Calls `setDataset` from class `SVHN_Detection`. 
        Prints which datasets have been successfully loaded.
        ### Parameter
        1. `title`: str, (default "Open Files") 
                - Title for filedialog.

        ### Catches
        - all exceptions and display their message in a messagebox.
        """
        f = fd.askopenfilenames(title=title,
                               initialdir=getcwd(),
                               filetypes=self.filetypes)
        
        try:
            train_loaded, val_loaded = self.svhn.setDataset(f)
        except Exception as e:
            mb.showwarning(title="Warning",
                           message="{}".format(e))
        else:
            if train_loaded:
                print("Train dataset sccuessfully loaded.")
            else:
                print("Train dataset not loaded.")
            
            if val_loaded:
                print("Validation dataset successfully loaded.")
            else:
                print("Validation dataset not loaded.")
            
    def __setEpochs(self):
        """Calls `setNumEpochs` from class `SVHN_Detection` to set the number of epochs for training.
        Reads the input from spinbox and converts it to an integer value before passing it to `setNumEpochs`.
        """
        e = int(self.sb_epochs.get())
        self.svhn.setNumEpochs(e)

    def __saveToDir(self):
        """Saves the mdoel to a specified directory.
        Opens a filedialog requesting a directory to which the model will be saved.
        This method returns, if the filedialog is closed without pressing "OK".
        If save was successful, it will be confirmed via messagebox.
        ### Catches 
        - all exceptions and display errormessage in a messagebox.
        """
        save_dir = fd.askdirectory(title="Save model", initialdir=getcwd(), mustexist=False)
        if len(save_dir) == 0:
            return
        try:
            self.svhn.saveModel(save_dir)
        except Exception as e:
            mb.showerror(title="Error!", 
                         message="Saving model to the specified path failed!\nException: {}".format(e))
        else:
            mb.showinfo(title="Success!", 
                        message="Successfully saved model.")
        
    def __loadFromDir(self):
        """Load a previously saved model into memory.
        A filedialog will ask for a directory to load the model from.
        `loadModelFromSave' from class `SVHN_Detection` is called with `compile`= True.
        The loaded model can then be used for prediction.
        """
        load_dir = fd.askdirectory(title="Load model", initialdir=getcwd(), mustexist=True)
        self.svhn.loadModelFromSave(load_dir, compile=True)

    def __askSave(self):
        """ 
        Asks the user to save the model if changes were made, otherwise the window will close 
        without displaying a messagebox. This method will only be called when closing the GUI.
        Prevents accidentally deleting all progress.
        """
        model_changed = self.svhn.getModelChanged()

        if model_changed:
            c = mb.askyesnocancel(title="Save changes?", 
                                  message="Do you want to save your changes?")
            if c == None:
                return
            elif c == True:
                self.__saveToDir()
                return
        
        self.window.destroy()
        return 
    
    def __menubarAbout(self):
        """ Method for the "About" menu option.
        Calls a messagebox displaying version, author and GitHub repository of this project.
        """
        mb.showinfo(title="About",
                    message="Version: {}\nAuthor: {}\nGitHub repository: {}".format(self.version, self.author, self.github_repo))
        
    def __mbVisitGitHub(self):
        """ Method for "Visit GitHub repo" menu option.
        Opens a new browser tab with the GitHub repository of this project.
        """
        web_open(self.github_repo, new=0)

    def __mbOpenHelp(self):
        """ Method for "Help" menu option.
        Opens README file of this project in browser.
        """
        web_open(self.readme, new=0)
        
    def __showImageGrid(self):
        """ Shows 25 random images of prediction.
        Calls `predictModel` from class `SVHN_Detection` with `validation_images` as parameter.
        If `predictModel` raises no exception `plotImageGrid` will be called with 
        previously calculated predictions.
        ### Catches
        - `AttributeError` - Displays error message in a messagebox.
        - `ValueError` - Displays a messagebox with "No validation data loaded!".
        """
        try:
            p = self.svhn.predictModel(self.svhn.validation_images)
        except AttributeError as e:
            mb.showwarning(title="Warning",
                           message="{}".format(e))
        except ValueError:
            mb.showwarning(title="No data",
                           message="No validation data loaded!")
        else:
            self.svhn.plotImageGrid(p)

    def __showLAGraph(self):
        """ Plots loss and accuracy over each epoch during training.
        Calls `plotLossAndAcc` from class `SVHN_Detection` with data collected during training.
        The plot will show two subplots with two graphs each. One graph represent loss or accuracy 
        from train dataset and the other one the validation 
        (validation datawhich were split off the train dataset).
        ### Catches
        - `AttributeError` - Displays error message in messagebox.
        """
        try:
            self.svhn.plotLossAndAcc(self.svhn.fit)
        except AttributeError as e:
            mb.showwarning(title="Warning",
                           message="{}".format(e))

if __name__ == "__main__":
    root = tk.Tk()
    my_gui = SVHN_GUI(root)
    root.mainloop()

