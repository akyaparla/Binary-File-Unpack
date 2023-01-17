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
        self.fileName = fileName.replace('\\', '/')
        
        self.offset = 0

        ## Parsing Metadata
        fileVersion = self.getDtype('int32', 4)
        self.fs:np.float32 = self.getDtype('float32', 4)
        devCount = self.getDtype('uint32', 4)

        DevID = np.empty(devCount, np.int32)
        SNL = np.empty(devCount, np.uint32)
        NameL = np.empty(devCount, np.uint32)
        NumEnChan = np.empty(devCount, np.uint32)
        # Lists to be reshaped into arrays later
        SN_lst = []
        Name_lst = []
        ChanNum_lst = []

        for i in range(devCount):
            DevID[i] = self.getDtype('int32', 4)
            
            SNL[i] = self.getDtype('uint32', 4)
            SN_lst.append(self.getDtype('int8', 1, SNL[i]))
            
            NameL[i] = self.getDtype('uint32', 4)
            Name_lst.append(self.getDtype('int8', 1, NameL[i]))

            NumEnChan[i] = self.getDtype('uint32', 4)
            ChanNum_lst.append(self.getDtype('int32', 4, NumEnChan[i]))

        # Converting Lists into Arrays
        SN = np.array(SN_lst)
        Name = np.array(Name_lst)
        chanNum = np.array(ChanNum_lst)

        # Names might not be in the order of the sensors, so get order of sensors
        self.num_sens:int = 2*devCount
        order_name = np.argsort(Name[:, -1])
        order = np.empty(self.num_sens, 'int')
        for i in range(len(order_name)):
            order[2*i] = 2*order_name[i]
            order[2*i+1] = 2*order_name[i]+1
        
        SN = SN[order_name, :]
        Name = Name[order_name, :]

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
                Tsec = self.getDtype('uint64', 8)
                TNsec = self.getDtype('uint32', 4)
                NS = self.getDtype('uint32', 4)
                Nt = NS // NumEnChan[i]
                d = self.getDtype('float32', 4, NumEnChan[i]*Nt).reshape((NumEnChan[i], Nt), order='F')
                if c >= self.data.shape[0]:
                    tmp = np.empty((c+Nt, NumEnChan[0], devCount))
                    tmp[:c, :, :] = self.data
                    self.data = tmp
                self.data[c:c+Nt, :, i] = d.T
            
            c += Nt
            status = BinaryFileUnpack.endOfFile(self)

        ## Getting Temperature and Pressure Data
        def temp_convert(temp) -> float:
            'Converts measured temp data from volts to deg Celsius'
            return (temp - 1.478) / 0.01746 + 25


        P = np.empty((self.num_sens, self.data.shape[0]))
        T = np.empty((self.num_sens, self.data.shape[0]))
        for i in range(devCount):
            T[2*i]   = temp_convert(self.data[:, 1, i])
            T[2*i+1] = temp_convert(self.data[:, 3, i])
            P[2*i]   = self.data[:, 0, i]
            P[2*i+1] = self.data[:, 2, i]

        self.P = P[order]
        self.T = T[order]
        # Creating time series
        self.time = np.linspace(0, (self.P.shape[1])/self.fs, self.P.shape[1])

        ## Apply pressure and temperature corrections by serial number
        # Numpy array with all the sensor corrections, first column is serial number, 
        # second column is P correction, third column is P correction std. dev.
        # fourth column is T correction (volts), fifth column is T correction std. dev.
        sens_corr = np.array([
            [5122769, -0.0047, 0.0029,  0.0147, 0.0091],
            [5122770,  0.0007, 0.0024, -0.0044, 0.0185],
            [5122777,  0.0034, 0.0104, -0.0013, 0.0090],
            [5122778,  0.0182, 0.0046, -0.0015, 0.0102],
            [5940428, -0.0046, 0.0031,  0.0019,	0.0089],
            [5940430, -0.0131, 0.0031, -0.0094, 0.0133],
        ])

        # Numpy array with sensor SN with index corresponding to position
        sens_used = np.array([5122778, 5122770, 5940428, 5122769, 5122777, 5940430])

        # Apply temperature and pressure corrections
        for i in range(len(sens_used)):
            # See if correction can be applied to the sensor
            ind_arr = np.where(sens_corr[:, 0] == sens_used[i])[0]
            if len(ind_arr) > 0:
                j = ind_arr[0]
                self.P[i] += sens_corr[j, 1]
                self.T[i] += temp_convert(sens_corr[j, 3] + 1.478) - 25

    
    def spectra(self, data_spec: np.ndarray, freq_type:str='pgram', freq_range: type=None) -> np.ndarray:
        '''
        Returns the frequencies (in Hertz) and the power (in deciBels) present in the spectrum of the data. Utilizes the scipy module to return the spectrum.

        Parameters:
            data_spec (ndarray): Frequencies contatining the data for each sensor. The array is 2-dimensional: the first axis is the sensor index and the second axis is the data.
            freq_type (str): Chooses between the options ('pgram', 'lombscargle'). Periodogram is recommended for eruption data, Lomb-Scargle is recommended for oscillation data.
            freq_range (tuple[float]): Range of desired frequencies. Default is (0.01, self.fs//2). The first element has to be greater than 0, while the second has to be less 
                    than self.fs//2, the Nyquist bound. Any frequencies outside this range will be converted to the default.
        Returns:
            Pxx (ndarray): The resulting 3-dimensional array storing frequency and power data of the complete spectrum.
                First axis: Index 0 is frequency data, index 1 is power data.
                Second axis: Contains the sensor indices.
                Third axis: The data of either frequency or power for the respective sensor.
        '''
        # Getting Frequencies for Spectral Analysis
        import scipy.signal as signal
        # Length of first axis indicates number of sensors
        # Length of second axis is number of data points

        # Reshape array to 2D if one sensor is available
        if len(data_spec.shape) == 1:
            data_spec = np.reshape(data_spec, (1, len(data_spec)))

        spec_sens = data_spec.shape[0]
        N = data_spec.shape[1]

        # Lomb-Scargle Approach
        length = 5000
        # Range of angular frequencies
        w = np.linspace(2*np.pi*0.01, 2*np.pi*self.fs/2, length)
        if freq_range is not None:
            # Correcting out of bounds freqs
            if freq_range[0] <= 0:
                freq_range[0] = 0.01
            if freq_range[1] >= self.fs/2:
                freq_range[1] = self.fs/2
            w = np.linspace(2*np.pi*freq_range[0], 2*np.pi*freq_range[1], length)

        Pxx = np.empty((2, spec_sens, length))
        for i in range(spec_sens):
            pgram = signal.lombscargle(
                self.time[:N], data_spec[i], freqs=w, precenter=True)
            power = 10 * np.log10(pgram)
            Pxx[:, i, :] = np.stack([w/(2*np.pi), power])

        # Periodogram approach
        # Pxx = np.empty((2, spec_sens, N//2 + 1))
        # for i in range(spec_sens):
        #     f, Pper_spec = signal.periodogram(data_spec[i], self.fs, 'cosine', scaling='density')
        #     power = 10 * np.log10(Pper_spec)
        #     Pxx[:, i, :] = np.stack([f, power])

        return Pxx

    def plot_static(self, x:np.ndarray, y:np.ndarray, x_label:str, y_label:str, plots_shape:tuple, color:str='blue', 
                    x_axis_type:str='linear', x_range:tuple=None, sharex=True, plot_type='line', figfilename:str=None):
        '''
        Outputs a static plot of the sensor data against a time series. Utilizes the matplotlib module.

        Parameters:
            x (ndarray): A (1-D) array of the time series or (2-D) array of the frequency or measurement data. If 2-D, axis convention follows that of the param y.
            y (ndarray): A 2-dimensional array with the sensor data or power spectrum. 
                First axis: The sensor index.
                Second axis: Contains the data for that sensor.
            x_label (str): The name of the data that defines the x-axis.
            y_label (str): The type of sensor data along with its units.
            plots_shape (tuple): The shape of the plots in the output in terms of number of rows and columns. Input should be (rows, cols).
            color (Any): Identifies the line_color feature of each line glyph for the plots. Allowed inputs are those allowed by line_color (default is element in bokeh.palettes.Turbo6).
            x_axis_type ('linear', 'log'): The scale of the x-axis, either can be a linear axis (='linear;) or logarithmic axis (='log'). Default is 'linear'.
            x_range (tuple): The lower and upper bounds of the x-axis.
            sharex (bool): If true, the plots will share the same x-scale. Otherwise, they will have independent scales.
            plot_type ('line', 'scatter'): Defines the type of plot displayed. Default is 'line'.
        
        Raises:
            ValueError:
                The plot dimension does not match the number of plots (same as the number of sensors).
        '''
        import matplotlib.pyplot as plt

        if plots_shape[0]*plots_shape[1] != y.shape[0]:
            raise ValueError(f"Plot dimension does not match number of plots. Plot Dimension: {plots_shape}, Number of plots: {y.shape[0]}")

        fig, ax = plt.subplots(plots_shape[0], plots_shape[1], sharex=sharex, figsize=(10, 5/3*self.num_sens), tight_layout=True)
        fig.text(0.5, -0.015, x_label, ha='center')
        fig.text(-0.015, 0.5, y_label, va='center', rotation='vertical')

        if plots_shape[0] == 1:
            ax = np.reshape(ax, (1, plots_shape[1]))
        elif plots_shape[1] == 1:
            ax = np.reshape(ax, (plots_shape[0], 1))
        
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                if plot_type == 'line':
                    ax[i][j].plot(x[ax.shape[1]*i+j] if x.shape[0] == y.shape[0] else x, y[ax.shape[1]*i+j], color=color)
                elif plot_type == 'scatter':
                    ax[i][j].scatter(x[ax.shape[1]*i+j] if x.shape[0] == y.shape[0] else x, y[ax.shape[1]*i+j], color=color, s=3)
                ax[i][j].set_xscale(x_axis_type)
                ax[i][j].set_title(f"Sensor {ax.shape[1]*i+j+1}")
                if x_range is not None:
                    if len(x_range) != 2: raise IndexError(f"x_range should contain only 2 values, has {len(x_range)}.")
                    ax[i][j].set_xlim(x_range)

        if figfilename is not None:
            fig.savefig(figfilename,transparent=True)

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
            x_axis_type ('linear', 'log'): The scale of the x-axis, either can be a linear axis (='linear;) or logarithmic axis (='log'). Default is 'linear'. 
            output_format ('file', 'notebook'): Outputting a file in either a .html file (='file') or within the notebook (='notebook'). Default is 'file'.
        
        Raises:
            ValueError: 
                output_format is not either 'file' or 'notebook'.
                The plot dimension does not match the number of plots (same as the number of sensors).
        '''
        from bokeh.plotting import figure, output_file, output_notebook, reset_output, show, ColumnDataSource
        from bokeh.layouts import gridplot, column
        from bokeh.models import CustomJS
        from bokeh.models.tools import HoverTool, CrosshairTool, BoxZoomTool, WheelZoomTool, SaveTool, ResetTool
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

        # Adjust the y-axis for each plot on change of time_range
        time_range.js_on_change('value', CustomJS(
            args=dict(plots=plots, sources=sources), 
            code="""
            // Getting x-data (shared between all sources)
            var x_arr = sources[0].data['x'];
            var x_range = cb_obj.value;
            // Getting indices of range slider
            var start_ind = x_arr.indexOf(x_range[0]);
            var end_ind = x_arr.indexOf(x_range[1]);
            if (start_ind == -1) {
                start_ind = 0;
            }
            if (end_ind == -1) {
                end_ind = x_arr.length - 1;
            }
            // Iterate through all the sensors
            for (let i = 0; i < plots.length; i++) {
                // Sensor number
                var sensNum = 'Sensor '+(i+1);
                
                // Getting y-data
                var y_arr = sources[i].data[sensNum];
                var y_range_arr = y_arr.slice(start_ind, end_ind);
                
                // Getting max, min, and range of data
                var max = Math.max(...y_range_arr.filter((el) => !isNaN(el)));
                var min = Math.min(...y_range_arr.filter((el) => !isNaN(el)));
                var range = max - min;
                
                // Dynamically changing the axis range
                plots[i].y_range.start = min - range/25;
                plots[i].y_range.end = max + range/25;
            }
            """
            ))

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