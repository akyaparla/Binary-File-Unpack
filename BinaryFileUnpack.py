import numpy as np
class BinaryFileUnpack:
    '''
    Class to neatly unpack binary files. Provides the header information as well as the data for the sensors.

    Methods:
        __init__: Method to initialize a BinaryFileUnpack object.
        spectra (np.ndarray): Returns the frequencies (in Hertz) and the power (in deciBels) present in the spectrum of the data. Utilizes the scipy module to return the spectrum.
        plot_static: Outputs a static plot of the sensor data against a time series. Utilizes the matplotlib module.
        plot_interactive: Outputs a plot of the sensor data against a time series. Implements a HoverTool and a CrossHairTool for the user to analyze the data.
        getDtype: Method to read the bytes of a file as a given data type. Main purpose is to be used in __init__ method.
        endOfFile (bool): Checks whether the end of file has been reached. Main purpose is to be used in __init__ method.
    '''
    def __init__(self, fileName:str):
        '''
        Method to initialize a BinaryFileUnpack object.

        Parameters:
            fileName: The file path for the binary file to be analyzed

        Instance variables:
            fileName (str): File path of the input binary file.
            fs (float): The number of samples taken per second (sampling rate). Measured in Hertz.
            header_info (dict: [str, Any]): Header info from parsing the metadata of the binary file
                Keys: Description of the data.
                Values: Value of the data (can be int, float, or ndarray depending on the data).
            num_sens (int): Number of sensors recording data.
            P (ndarray): A 2-dimensional array that returns the pressure data for each sensor.
                First axis: The sensor indices (0 to self.num_sens-1, inclusive).
                Second axis: The pressure data for the respective sensor for a time series (given by self.time).
            T (ndarray): A 2-dimensional array that returns the temperature data for each sensor.
                First axis: The sensor indices (0 to self.num_sens-1, inclusive).
                Second axis: The temperature data for the respective sensor for a time series (given by self.time).
            time (ndarray): The time series for the data given in self.P and self.T.
            offset: A variable to track the number of bytes transversed in the binary file. Primarily meant for internal use.
        '''
        self.fileName = fileName
        
        self.offset = 0

        ## Parsing Metadata
        fileVersion = BinaryFileUnpack.getDtype(self, 'int32', 4)
        self.fs = BinaryFileUnpack.getDtype(self, 'float32', 4)
        devCount = BinaryFileUnpack.getDtype(self, 'uint32', 4)

        DevID = np.empty(devCount, np.int32)
        SNL = np.empty(devCount, np.uint32)
        NameL = np.empty(devCount, np.uint32)
        NumEnChan = np.empty(devCount, np.uint32)
        # Lists to be reshaped into arrays later
        SN_lst = []
        Name_lst = []
        ChanNum_lst = []

        for i in range(devCount):
            DevID[i] = BinaryFileUnpack.getDtype(self, 'int32', 4)
            
            SNL[i] = BinaryFileUnpack.getDtype(self, 'uint32', 4)
            SN_lst.append(BinaryFileUnpack.getDtype(self, 'byte', 1, SNL[i]))
            
            
            NameL[i] = BinaryFileUnpack.getDtype(self, 'uint32', 4)
            Name_lst.append(BinaryFileUnpack.getDtype(self, 'byte', 1, NameL[i]))

            NumEnChan[i] = BinaryFileUnpack.getDtype(self, 'uint32', 4)
            ChanNum_lst.append(BinaryFileUnpack.getDtype(self, 'int32', 4, NumEnChan[i]))

        # Converting Lists into Arrays
        SN = np.array(SN_lst)
        Name = np.array(Name_lst)
        chanNum = np.array(ChanNum_lst)
        
        # Store all header information in dictionary
        self.header_info = {
            "File Version": fileVersion, "Sampling Rate": self.fs,
            "Device Count": devCount, "Device ID": DevID,
            "Serial Number Length": SNL, "Serial Number": SN,
            "Name Length": NameL, "Name": Name,
            "Number of Enabled Channels": NumEnChan, "Channel Number": chanNum            
        }

        ## Parsing data
        status = False  # EOF marker
        c = 0
        Trel = 0
        self.data = np.empty((10000, NumEnChan[0], devCount))

        while not status:
            for i in range(devCount):
                Tsec = BinaryFileUnpack.getDtype(self, 'uint64', 8)
                TNsec = BinaryFileUnpack.getDtype(self, 'uint32', 4)
                NS = BinaryFileUnpack.getDtype(self, 'uint32', 4)
                Nt = NS // NumEnChan[i]
                d = BinaryFileUnpack.getDtype(self, 'float32', 4, NumEnChan[i]*Nt).reshape((NumEnChan[i], Nt), order='F')
                if c >= self.data.shape[0]:
                    tmp = np.empty((c+Nt, NumEnChan[0], devCount))
                    tmp[:c, :, :] = self.data
                    self.data = tmp
                self.data[c:c+Nt, :, i] = d.T
            
            c += Nt
            status = BinaryFileUnpack.endOfFile(self)

        ## Getting Temperature and Pressure Data
        self.num_sens:int = 2*devCount
        self.T = np.empty((self.num_sens, self.data.shape[0]))
        self.P = np.empty((self.num_sens, self.data.shape[0]))
        for i in range(devCount):
            self.T[2*i]   = (self.data[:, 1, i] - 1.478) / 0.01746 + 25
            self.T[2*i+1] = (self.data[:, 3, i] - 1.478) / 0.01746 + 25
            self.P[2*i]   = self.data[:, 0, i]
            self.P[2*i+1] = self.data[:, 2, i]

        # Creating time series
        self.time = np.arange(0, (len(self.T[0, :]))/self.fs, 1/self.fs)
    
    def spectra(self, data_spec:np.ndarray) -> np.ndarray:
        '''
        Returns the frequencies (in Hertz) and the power (in deciBels) present in the spectrum of the data. Utilizes the scipy module to return the spectrum.

        Parameters:
            data_spec (ndarray): Frequencies contatining the data for each sensor. The array is 2-dimensional: the first axis is the sensor index and the second axis is the data.
        
        Returns:
            Pxx (ndarray): The resulting 3-dimensional array storing frequency and power data of the complete spectrum.
                First axis: Index 0 is frequency data, index 1 is power data.
                Second axis: Contains the sensor indices.
                Third axis: The data of either frequency or power for the respective sensor.
        '''
        # Getting Frequencies for Spectral Analysis
        import scipy.signal as signal
        N = self.data.shape[0]
        if self.num_sens == 1:
            data_spec = np.reshape(data_spec, (1, len(data_spec)))
            
        Pxx = np.empty((2, self.num_sens, int(np.ceil(N/2))))
        for i in range(self.num_sens):
            f, Pper_spec = signal.periodogram(data_spec[i], self.fs, 'cosine', scaling='density')
            power = 10 * np.log10(Pper_spec)
            Pxx[:,i, :] = np.stack([f, power])
        return Pxx

    def plot_static(self, x:np.ndarray, y:np.ndarray, x_label:str, y_label:str, 
                    plots_shape:tuple, color:str='blue', x_axis_type:str='linear'):
        '''
        Outputs a static plot of the sensor data against a time series. Utilizes the matplotlib module.

        Parameters:
            x (ndarray): A (1-D) array of the time series or (2-D) array of the frequency data. If 2-D, axis convention follows that of the param y.
            y (ndarray): A 2-dimensional array with the sensor data or power spectrum. 
                First axis: The sensor index.
                Second axis: Contains the data for that sensor.
            x_label (str): The name of the data that defines the x-axis.
            y_label (str): The type of sensor data along with its units.
            plots_shape (tuple): The shape of the plots in the output in terms of number of rows and columns. Input should be (rows, cols).
            color (Any): Identifies the line_color feature of each line glyph for the plots. Allowed inputs are those allowed by line_color (default is element in bokeh.palettes.Turbo6).
            x_axis_type ('linear', 'log'): The scale of the x-axis, either can be a linear axis (='linear;) or logarithmic axis (='log'). Default is 'linear'. 'log' option has bugs with the x-axis labels due to log(0) errors.                 
        
        Raises:
            ValueError:
                The plot dimension does not match the number of plots (same as the number of sensors).
        '''
        import matplotlib.pyplot as plt

        if plots_shape[0]*plots_shape[1] != y.shape[0]:
            raise ValueError(f"Plot dimension does not match number of plots. Plot Dimension: {plots_shape}, Number of plots: {self.num_sens}")

        fig, ax = plt.subplots(plots_shape[0], plots_shape[1], sharex=True, figsize=(10, 5/3*self.num_sens), tight_layout=True)
        fig.text(0.5, -0.015, x_label, ha='center')
        fig.text(-0.015, 0.5, y_label, va='center', rotation='vertical')

        if plots_shape[0] == 1:
            ax = np.reshape(ax, (1, plots_shape[1]))
        elif plots_shape[1] == 1:
            ax = np.reshape(ax, (plots_shape[0], 1))
        
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i][j].plot(x[ax.shape[1]*i+j] if x.shape[0] == y.shape[0] else x, y[ax.shape[1]*i+j], color=color)
                ax[i][j].set_xscale(x_axis_type)
                ax[i][j].set_title(f"Sensor {ax.shape[1]*i+j+1}")

    def plot_interactive(self, x:np.ndarray, y:np.ndarray, x_label:str, y_label:str, 
                         plots_shape:tuple, color=None, x_axis_type:str='linear', output_format:str='file'):
        '''
        Outputs a plot of the sensor data against a time series. Implements a HoverTool and a CrossHairTool for the user to analyze the data.

        Parameters:
            x (ndarray): A (1-D) array of the time series or (2-D) array of the frequency data. If 2-D, axis convention follows that of the param y.
            y (ndarray): A 2-dimensional array with the sensor data or power spectrum. 
                First axis: The sensor index.
                Second axis: Contains the data for that sensor.
            x_label (str): The name of the data that defines the x-axis.
            y_label (str): The type of sensor data along with its units.
            plots_shape (tuple): The shape of the plots in the output in terms of number of rows and columns. Input should be (rows, cols).
            color (Any): Identifies the line_color feature of each line glyph for the plots. Allowed inputs are those allowed by line_color (default is element in bokeh.palettes.Turbo6).
            x_axis_type ('linear', 'log'): The scale of the x-axis, either can be a linear axis (='linear;) or logarithmic axis (='log'). Default is 'linear'. 'log' option has bugs with the x-axis labels due to log(0) errors.                 
            output_format ('file', 'notebook'): Outputting a file in either a .html file (='file') or within the notebook (='notebook'). Default is 'file'.
        
        Raises:
            ValueError: 
                output_format is not either 'file' or 'notebook'.
                The plot dimension does not match the number of plots (same as the number of sensors).
        '''
        from bokeh.plotting import figure, output_file, output_notebook, reset_output, show, ColumnDataSource
        from bokeh.models.tools import HoverTool, CrosshairTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
        from bokeh.layouts import gridplot, column
        from bokeh.models.widgets import RangeSlider
        
        if plots_shape[0]*plots_shape[1] != y.shape[0]:
            raise ValueError(f"Plot dimension does not match number of plots. Plot Dimension: {plots_shape}, Number of plots: {self.num_sens}")

        # Initialize the plots into a numpy array
        plots = np.empty(self.num_sens, dtype='object')
        for i in range(self.num_sens):
            # Determine the x_axis range
            x_to_use = x if len(x.shape)==1 else x[i]
            x_range = (x_to_use[0], x_to_use[-1])
            
            # Logarithm Case - make sure there are no zeros involved
            if x_axis_type == 'log':
                if x_range[0] <= 0:
                    x_range = (x_to_use[1], x_range[1])
                
                x_range = (10**np.floor(np.log10(x_range[0])), 10**np.ceil(np.log10(x_range[1])))

            plots[i] = figure (
                title = f"Sensor {i+1}",
                sizing_mode = "stretch_width",
                plot_height = 675//plots_shape[0],
                x_range=x_range,
                x_axis_type=x_axis_type,
            )

        # Create ColumnDataSources with x and y data for each sensor
        sources = np.empty(self.num_sens, dtype=ColumnDataSource)
        for i in range(self.num_sens):
            sources[i] = ColumnDataSource({'x': x[i] if x.shape[0] == y.shape[0] else x , f'Sensor {i+1}':y[i]})

        # Double-ended range slider so user can select data in a given time frame
        x_to_use = x if len(x.shape)==1 else x[0]
        time_range = RangeSlider(start=x_to_use[1], end=x_to_use[-1], value=(x_to_use[1], x_to_use[-1]), step=0.01, sizing_mode="stretch_width")
        for plot in plots:
            time_range.js_link('value', plot.x_range, 'start', attr_selector=0)
            time_range.js_link('value', plot.x_range, 'end', attr_selector=1)

        if color is None:
            from bokeh.palettes import Turbo6
            import random 
            color = random.choice(Turbo6)

        hover_tools = np.empty(self.num_sens, dtype=object) 
        for i in range(self.num_sens):
            hover_tools[i] = HoverTool(
                tooltips = [
                    (x_label, f"@x"),
                    (y_label, f"@{{Sensor {i+1}}}")
                ],
                renderers = [plots[i].line(x='x', y=f"Sensor {i+1}", line_color = color, source=sources[i])]
            )
            plots[i].tools = [
                hover_tools[i], 
                CrosshairTool(dimensions="height", line_alpha=0.5, line_color="purple"), 
                BoxZoomTool(), WheelZoomTool(), SaveTool(), ResetTool()
            ]

        # Adding axis labels
        grid_layout = np.reshape(plots, plots_shape)
        # x-axis
        for i in range(grid_layout.shape[1]):
            grid_layout[-1][i].xaxis.axis_label = x_label
        # y-axis
        for i in range(grid_layout.shape[0]):
            grid_layout[i][0].yaxis.axis_label = y_label

        # Formatting the Graphs
        grid_plots = gridplot(list(grid_layout), sizing_mode = "stretch_width")
        grid = column(grid_plots, time_range, sizing_mode="stretch_width")

        reset_output()
        if output_format == 'file':
            y_strip = y_label.strip(' \\><:\"|?*$')
            x_strip = x_label.strip(' \\><:\"|?*$')
            output_file(f"{self.fileName.split('.')[0]}-{y_strip}-{x_strip}.html")
        elif output_format == 'notebook':
            output_notebook()
        else:
            raise ValueError(f"{output_format} is not a valid option for output_format. Allowed options are 'file' and 'notebook'")
        show(grid)

    def getDtype(self, type, numByte:int, size:int=1):
        '''
        Method to read the bytes of a file as a given data type. Main purpose is to be used in __init__ method.

        Parameters:
            type: Type of data type to read. Examples include 'float32', np.int8, etc.
            numByte: The number of bytes for 1 unit of the type. For example, 'float32' reads 4 bytes at a time, and np.int8 reads 1 byte at a time.
            size (int): The number of the inputted data type to read. Default is 1.

        Returns:
            Either an object of type np.ndarray or the indicated data type (param type) containing said data types read from the file.
        '''
        tmp = self.offset
        self.offset += numByte*size
        if size == 1:
            return np.fromfile(self.fileName, dtype=type, offset=tmp)[0]
        return np.fromfile(self.fileName, dtype=type, offset=tmp)[:size]

    def endOfFile(self) -> bool:
        '''
        Checks whether the end of file has been reached. Main purpose is to be used in __init__ method.

        Returns:
            (bool) True if end of file is reached; otherwise False.
        '''
        return self.offset == len(np.fromfile(self.fileName, dtype='byte'))
