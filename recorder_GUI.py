from tkinter import *
from notebook import *   # window with tabs
from tkinter import filedialog, messagebox
from AnalyzeRecording import analyzer



class recorder_GUI:
    def __init__(self, parent):
        self.parent = parent
        self.initUI()

    def initUI(self):
        row = 0
        choose_label = "Input folder:"
        Label(self.parent, text=choose_label).grid(row=row, column=0, sticky=W, padx=5, pady=(10, 2))
        row = row + 1
        #
        # TEXTBOX TO PRINT PATH OF THE SOUND FILE
        self.filelocation = Entry(self.parent)
        self.filelocation.focus_set()
        self.filelocation["width"] = 25
        self.filelocation.grid(row=row, column=0, sticky=W, padx=10)
        self.filelocation.delete(0, END)
        self.filelocation.insert(0, './resources')
        #row = row + 1

        # BUTTON TO BROWSE SOUND FILE
        self.open_file = Button(self.parent, text="Browse...", command=self.browse_file)  # see: def browse_file(self)
        self.open_file.grid(row=row, column=0, sticky=W, padx=(250, 6))  # put it beside the filelocation textbox
        row = row + 1
        #
        # # BUTTON TO PREVIEW SOUND FILE
        # #self.preview = Button(self.parent, text=">", command=lambda: UF.wavplay(self.filelocation.get()), bg="gray30",
        # #                      fg="white")
        # #self.preview.grid(row=1, column=0, sticky=W, padx=(306, 6))
        #


        # BUTTON TO COMPUTE EVERYTHING
        self.Gstring = Button(self.parent, text="G-string", command=self.record(4), bg="dark red", fg="white")
        self.Gstring.grid(row=row, column=0, padx=5, pady=(10, 2), sticky=W)
        row = row + 1
        self.Dstring = Button(self.parent, text="D-string", command=self.record(3), bg="dark red", fg="white")
        self.Dstring.grid(row=row, column=0, padx=5, pady=(10, 2), sticky=W)
        row = row + 1
        self.Astring = Button(self.parent, text="A-string", command=self.record(3), bg="dark red", fg="white")
        self.Astring.grid(row=row, column=0, padx=5, pady=(10, 2), sticky=W)
        row = row + 1
        self.Estring = Button(self.parent, text="E-string", command=self.record(3), bg="dark red", fg="white")
        self.Estring.grid(row=row, column=0, padx=5, pady=(10, 2), sticky=W)
        row = row + 1

        # # BUTTON TO PLAY OUTPUT
        # output_label = "Output:"
        # Label(self.parent, text=output_label).grid(row=13, column=0, sticky=W, padx=5, pady=(10, 15))
        # self.output = Button(self.parent, text=">", command=lambda: UF.wavplay(
        #     'output_sounds/' + os.path.basename(self.filelocation.get())[:-4] + '_harmonicModel.wav'), bg="gray30",
        #                      fg="white")
        # self.output.grid(row=13, column=0, padx=(60, 5), pady=(10, 15), sticky=W)

        # define options for opening file
        self.file_opt = options = {}
        options['defaultextension'] = '.wav'
        options['filetypes'] = [('All files', '.*'), ('Wav files', '.wav')]
        options['initialdir'] = '../../sounds/'
        options['title'] = 'Open a mono audio file .wav with sample frequency 44100 Hz'


    def browse_file(self):
        self.filename = filedialog.askopenfilename(**self.file_opt)

        # set the text of the self.filelocation
        self.filelocation.delete(0, END)
        self.filelocation.insert(0, self.filename)


    def record(self, string):
        #instantiate the recorder class
        return

class analyzer_GUI:
    def __init__(self, parent):
        self.parent = parent
        self.analyzer= analyzer()
        self.initUI()

    def initUI(self):

        # BUTTON TO COMPUTE EVERYTHING
        row=0
        self.analysis = Button(self.parent, text="Analyze", command=self.analyzer.analyze, bg="dark red", fg="white")
        self.analysis.grid(row=row, column=0, padx=5, pady=(10, 2), sticky=W)





if __name__=="__main__":
    root = Tk()
    root.title('Violin Trainer GUI')
    nb = notebook(root, TOP)  # make a few diverse frames (panels), each using the NB as 'master':

    # uses the notebook's frame
    f1 = Frame(nb())
    recorder = recorder_GUI(f1)
    f2 = Frame(nb())
    analyzer = analyzer_GUI(f2)
    f3 = Frame(nb())
    analyzer = recorder_GUI(f3)

    nb.add_screen(f1, "Record")
    nb.display(f1)
    nb.add_screen(f2, "Analyze")
    nb.display(f2)
    nb.add_screen(f3, "Train")
    nb.display(f3)
    root.geometry('+0+0')
    root.mainloop()


    print("DONE")