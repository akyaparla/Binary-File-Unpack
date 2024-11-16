import numpy as np
import csv
import warnings
class BinaryFileUnpack:
    '''
    Class to neatly unpack binary files. Provides the header information as well as the data for the sensors.

    Methods:
        __init__: Method to initialize a BinaryFileUnpack object.
        spectra (np.ndarray): Returns a periodogram of the data.
        spectrogram (np.ndarray): Returns the frequencies (in Hertz) and the power (in deciBels) as a function of a time series.
        plot_static: Outputs a static plot of the sensor data against a time series. Utilizes the Matplotlib module.
        plot_interactive (depreciated): Outputs a plot of the sensor data against a time series. Implements a HoverTool and a CrossHairTool for the user to analyze the data. Utilizes the Bokeh module.
        plot_eruption_PT: Outputs a Pressure-Temperature plot of an eruption. Utilizes the matplotlib module.
        getTimeRange: Returns a time series of the data. Intended for use as a time series input for other relavent methods.
        getDtype (np.ndarray or Object): Method to read the bytes of a file as a given data type. Main purpose is to be used in __init__ method.
        endOfFile (bool): Checks whether the end of file has been reached. Main purpose is to be used in __init__ method.
    '''
    def __init__(self, fileName:str, cut_range:tuple=None, sens_from_header:bool=True, new_file:bool=False):
        '''
        Method to initialize a BinaryFileUnpack object.

        Parameters:
            fileName (str): The file path for the binary file to be analyzed.
            cut_range (tuple): The start and end times, in seconds, to cut the file. Default is None; file is not cut.
            sens_from_header (bool): Selects Sensor SN from header csv if selected, prestored if False. Default is True.
            new_file (bool): If cut_range is provided, make new .bin data and .csv header files. Default is False.

        Instance variables:
            fileName (str): File path of the input binary file.
            fs (float): The number of samples taken per second (sampling rate). Measured in Hertz.
            start_splice (float): Indicates the starting time in seconds of a spliced file.
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
            Pstd (ndarray): A 1-dimensional array containing the standard deviations of the Pressure calibration for each sensor.
            time (ndarray): The time series for the data given in self.P and self.T.
            offset (int): A variable to track the number of bytes transversed in the binary file. Primarily meant for internal use.
        
        Raises:
            Warning: Header file not Detected, will use default sensor positions.
        '''

        self.fileName = fileName.replace('\\', '/')
        self.offset = 0

        ## Parsing Metadata
        with open(fileName, "rb") as file:
            self.fileContent = file.read()   

        fileVersion = self.getDtype('i', 4)
        self.fs = self.getDtype('f', 4)
        devCount = self.getDtype('I', 4)

        DevID = np.empty(devCount, np.int32)
        SNL = np.empty(devCount, np.uint32)
        NameL = np.empty(devCount, np.uint32)
        NumEnChan = np.empty(devCount, np.uint32)
        # Lists to be reshaped into arrays later
        SN_lst = []
        Name_lst = []
        ChanNum_lst = []
        for i in range(devCount):
            DevID[i] = self.getDtype('i', 4)
            
            SNL[i] = self.getDtype('I', 4)
            SN_lst.append(self.getDtype('b', 1, SNL[i]))
            
            NameL[i] = self.getDtype('I', 4)
            Name_lst.append(self.getDtype('b', 1, NameL[i]))

            NumEnChan[i] = self.getDtype('I', 4)
            ChanNum_lst.append(self.getDtype('i', 4, NumEnChan[i]))
        
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

        # See if header file indicates a splice from a previous file.
        self.start_splice = 0
        splice_line = ""
        try: 
            with open(fileName.split('.')[0] + "_headerInfo.csv", 'r') as fr:
                for line in fr:
                    splice_line = line.rstrip()
            if splice_line.split(',')[0] == "Start Splice":
                self.start_splice[1] = float(splice_line.split(',')[1])
        except:
            pass

        # Store all header information in dictionary
        self.header_info = {
            "File Version": fileVersion, "Sampling Rate": self.fs,
            "Device Count": devCount, "Device ID": DevID,
            "Serial Number Length": SNL, "Serial Number": SN,
            "Name Length": NameL, "Name": Name,
            "Number of Enabled Channels": NumEnChan, "Channel Number": chanNum,
            "Start of splice": self.start_splice            
        }

        meta_bytes = self.offset

        ## Parsing data
        status = False  # EOF marker
        c = 0
        Trel = 0
        self.data = np.empty((int(1e7), NumEnChan[0], devCount))
        NS_sum = 0
        while not status:
            # Want to make a new .bin file if cut_range is specified
            if cut_range is not None: 
                # Keep track of when to record data
                c_start = -1
                start_byteloc = -1
                # Make new .bin file, new header .csv file.
                if c >= cut_range[1]*self.fs:
                    if new_file:
                        # .bin file
                        new_fn = fileName.split('.')[0] + f'_{c_start//self.fs}s-{c//self.fs}s.bin'
                        with open(new_fn, "wb") as new_file:
                            new_file.write(
                                self.fileContent[np.hstack(
                                    np.arange(int(meta_bytes)),
                                    np.arange(start_byteloc, self.offset)
                                )]
                            )
                        # .csv file
                        try:
                            with open(fileName.split('.')[0] + "_headerInfo.csv", 'r') as fr:
                                headers = fr.readlines()
                            new_head_file = fileName.split('.')[0] + f'_{c_start//self.fs}s-{c//self.fs}s_headerInfo.csv'
                            with open(new_head_file, 'w') as fw:
                                fw.writelines(headers)
                                fw.write(f"Start Splice,{round(c_start/self.fs, 9)}")
                        except FileNotFoundError:
                            warnings.warn("Header file of original not detected, new header not created.")

                        print(f"Data from {round(c_start/self.fs, 3)}s to {round(c/self.fs, 3)}s ({round((c-c_start)/self.fs, 3)} seconds) have been written into {new_fn}!")
                    break

                for i in range(devCount):
                    Tsec = self.getDtype('Q', 8)
                    TNsec = self.getDtype('I', 4)
                    NS = self.getDtype('I', 4)
                    NS_sum+=NS
                    Nt = NS // NumEnChan[i]
                    d = self.getDtype('f', 4, NumEnChan[i]*Nt).reshape((Nt, NumEnChan[i]))
                    if c >= cut_range[0]*self.fs:
                        if c_start < 0:
                            c_start = c
                            start_byteloc = self.offset - (8+4+4+4*NumEnChan[i]*Nt)
                        if c >= self.data.shape[0]:
                            tmp = np.empty((self.data.shape[0]*10, NumEnChan[0], devCount))
                            tmp[:c, :, :] = self.data
                            self.data = tmp
                        self.data[c:c+Nt, :, i] = d
                c += Nt
                status = BinaryFileUnpack.endOfFile(self)

            else:
                for i in range(devCount):
                    Tsec = self.getDtype('Q', 8)
                    TNsec = self.getDtype('I', 4)
                    NS = self.getDtype('I', 4)
                    NS_sum+=NS
                    Nt = NS // NumEnChan[i]
                    d = self.getDtype('f', 4, NumEnChan[i]*Nt).reshape((Nt, NumEnChan[i]))
                    if c >= self.data.shape[0]:
                        tmp = np.empty((self.data.shape[0]*10, NumEnChan[0], devCount))
                        tmp[:c, :, :] = self.data
                        self.data = tmp
                    self.data[c:c+Nt, :, i] = d

            c += Nt
            status = BinaryFileUnpack.endOfFile(self)
        
        self.data = self.data[:c]

        ## Getting Temperature and Pressure Data
        def temp_convert(temp, SN) -> float:
            'Converts measured temp data from volts to deg Celsius'
            if SN == 5122773:
                V25 = 1.466
                slope = 0.01739
            elif SN == 5122770:
                V25 = 1.470
                slope = 0.01742
            elif SN == 5122769:
                V25 = 1.484
                slope = 0.01698
            elif SN == 5122776:
                V25 = 1.478
                slope = 0.01698
            elif SN == 5122777:
                V25 = 1.480
                slope = 0.1739
            elif SN == 5122778:
                V25 = 1.475
                slope = 0.01748
            elif SN == 5940428:
                V25 = 1.477
                slope = 0.01744
            elif SN == 5940430:
                V25 = 1.484
                slope = 0.01734
            elif SN == 5940434:
                V25 = 1.476
                slope = 0.01690
            elif SN == 5961388:
                V25 = 1.473
                slope = 0.01688
            elif SN == 5940436:
                V25 = 1.475
                slope = 0.01701
            elif SN == 5940432:
                V25 = 1.474
                slope = 0.01693
            elif SN == 5940431:
                V25 = 1.482
                slope = 0.01688
            
            return (temp - V25) / slope + 25

        # Numpy array with sensor SN with index corresponding to position
        sens_used = np.array([5122778, 5122769, 5940428, 5122770, 5122777, 5940430])
        sens_used = sens_used[:self.num_sens]
        if sens_from_header:
            try:
                headerName = fileName.split('.')[0] + "_headerInfo.csv"
                with open(headerName, 'r') as file: # Find file, go to except statement otherwise
                    reader = csv.reader(file)
                    newOrder = np.zeros(devCount, dtype=int)
                    for i, line in enumerate(reader):
                        if i == 13: # Read location of sensors using manual naming
                            for j in range(devCount):
                                newOrder[j] = int(line[j+1][-1]) - 1

                        if i == 14: # 14th line in header file is Pressure 1 channel
                            for j in range(devCount):
                                sens_used[newOrder[j]] = int(line[j+1])
                                
                        elif i == 41: # Read location of sensors using manual naming
                            for j in range(devCount):
                                newOrder[j] = int(line[j+1][-1]) - 1
                            
                        elif i == 42: # 42nd line in header file is Pressure 3 channel
                            for j in range(devCount):
                                sens_used[newOrder[j]] = int(line[j+1])
                
            except FileNotFoundError:
                # No header file, use default sens and send warning
                msg = "\n"
                for j in range(self.num_sens):
                    msg += f"\tPosition {j+1}: SN {sens_used[j]}\n"
                warnings.warn("\nNo header file detecting, using default sensor settings."+msg)
        self.sens_used = sens_used

        ### Change as of 06/19/2024 : Take calibrations at beginning of day and apply those.

        P = np.empty((self.num_sens, self.data.shape[0]))
        T = np.empty((self.num_sens, self.data.shape[0]))
        for i in range(devCount):
            T[2*i]   = temp_convert(self.data[:, 1, i], sens_used[order[2*i]])
            T[2*i+1] = temp_convert(self.data[:, 3, i], sens_used[order[2*i+1]])
            P[2*i]   = self.data[: , 0, i]
            P[2*i+1] = self.data[: , 2, i]

        self.P = P[order]
        self.T = T[order]

        # Creating time series
        self.time = np.linspace(0, (self.P.shape[1])/self.fs, self.P.shape[1]) + self.start_splice

        ## Apply pressure and temperature corrections by serial number
        # 06/17/2024: No we don't, now we do daily calibration
            # Maybe turn them on when rerunning old tests??? Idk rn they're turned off

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

        # self.Pstd contains the standard deviations of calibrations
                # self.P[i] += sens_corr[j, 1]
                # self.T[i] += temp_convert(sens_corr[j, 3] + 1.478) - 25

        # Calibration to the absolute pressure (constant offset)
        # first column is serial number
        # Second column is offset to absolute pressure
        # Third column are standard deviations of the offsets
        sens_abs = np.array([
            [5122778, 0.22586478, 0.00115527],
            [5122769, 0.24026248, 0.00064523],
            [5940428, 0.2364321 , 0.00155191],
            [5122770, 0.2425973 , 0.00104353],
            [5122777, 0.22593766, 0.00115591],
            [5940430, 0.25612138, 0.00100135],
        ])

        # Apply corrections to sensors
        # for i in range(len(sens_used)):
        #     # See if correction can be applied to the sensor
        #     ind_arr = np.where(sens_corr[:, 0] == sens_used[i])[0]
        #     if len(ind_arr) > 0:
        #         j = ind_arr[0]
        #         self.P[i] = self.P[i] - sens_abs[j, 1]
        #         self.Pstd[i] = sens_abs[j, 2]

    def spectra(self, data_spec: np.ndarray) -> np.ndarray:
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
        # Length of first axis indicates number of sensors
        # Length of second axis is number of data points

        # Reshape array to 2D if one sensor is available
        if len(data_spec.shape) == 1:
            data_spec = np.reshape(data_spec, (1, len(data_spec)))

        spec_sens = data_spec.shape[0]
        N = data_spec.shape[1]

        # Periodogram approach
        Pxx = np.empty((2, spec_sens, N//2 + 1))
        for i in range(spec_sens):
            f, Pper_spec = signal.periodogram(data_spec[i], self.fs, 'cosine', scaling='density')
            power = 10 * np.log10(Pper_spec)
            Pxx[:, i, :] = np.stack([f, power])

        return Pxx
    
    def spectrogram(self, data:np.ndarray, window:int=20, times:np.ndarray=None) -> np.ndarray:
        '''
        Returns spectogram of the data with indicated window length. Taken at every second of the data.

        Parameters:
            data (ndarray): 1-D array of data to analyze.
            window (int): Time interval to use for FFT at each second.
            times (ndarray): Time series used for the data. Default is entire time series.

        Returns:
            spec (ndarray): The resulting 3-dimensional array storing frequency and power data of the complete spectrum.

                First axis: Time series of data, at every second.

                Second axis: Frequency or Power data, obtained by scipy.fft.rfftfreq or scipy.fft.rfft, respectively.

                Third axis: 0 or 1, if 0, second axis is frequency data, if 1, second axis power data.
        Raises:
            ValueError: Window out of length or data and time series length do not match
        '''
        import scipy.fft as fft
        # FFT on Data
        if times is None:
            times = self.time
        if len(data) != len(times):
            raise ValueError("Data array and time series lengths do not match", data.shape, times.shape)
        
        num_samples = int(window*self.fs/2) + 1
        duration = int(times[-1] - times[0])
        if window > duration:
            raise ValueError("Window length is longer than time series duration", window, duration)
        
        spec = np.zeros((duration, num_samples, 2))
        spec[:, :, 0] = fft.rfftfreq(int(window*self.fs), d=1/self.fs)
        for t in range(1, duration+1):
            if t < window:
                data_to_use = data[: int(t * self.fs)]
            else:
                data_to_use = data[int((t - window) * self.fs) : int(t * self.fs)]
            new_spec = fft.rfft(data_to_use)
            db_spec = 10*np.log10(new_spec)
            if t < window:
                freq = fft.rfftfreq(data_to_use.shape[0], d=1/self.fs)
                j = 0
                for i in range(spec.shape[1]):
                    if spec[t-1][i][0] > freq[j]:
                        j += 1
                    spec[t-1][i][1] = db_spec[j]

            else:
                spec[t-1, :, 1] = db_spec
        return spec

    def plot_static(self, x:np.ndarray, y:np.ndarray, x_label:str, y_label:str, plots_shape:tuple, color:str='blue', 
                    y2:np.ndarray=None, y2_label:str=None, color2:str='red', x_axis_type:str='linear', ordering:np.ndarray=None,
                    times:np.ndarray=None, sharex=True, sharey=False, plot_type='line', figfilename:str=None):
        '''
        Outputs a static plot of the sensor data against a time series. Utilizes the matplotlib module.

        Parameters:
            x (ndarray): A (1-D) array of the time series or (2-D) array of the frequency or measurement data. If 2-D, axis convention follows that of the param y.
            y (ndarray): A 2-D array with the sensor data or power spectrum. 
                First axis: The sensor index.
                Second axis: Contains the data for that sensor.
            x_label (str): The name of the data that defines the x-axis.
            y_label (str): The type of sensor data along with its units.
            plots_shape (tuple): The shape of the plots in the output in terms of number of rows and columns. Input should be (rows, cols).
            color (Any): Identifies the line_color feature of each line glyph for the plots. Allowed inputs are those allowed by line_color. Default is 'blue'.
            y2 (ndarray): An optional second array that is of the same dimension as @param y. Will be plotted alongside y with an additional axis.
            y2_label (str): The type of secondary sensor data along with its units.
            color2 (str): The color of the optional second data array. Default is 'red'.
            x_axis_type ('linear', 'log'): The scale of the x-axis, either can be a linear axis (='linear') or logarithmic axis (='log'). Default is 'linear'.
            ordering (list): A custom ordering of the sensor plot data. Indicate the sensor index (starting from 0), not the number.
                            Will throw a AssertError if len(ordering) != y.shape[0] or max(ordering) != y.shape[0] - 1.
            times (ndarray): A time series for the data.
            sharex (bool): If true, the plots will share the same x-scale. Otherwise, they will have independent scales.
            sharey (bool): If true, the plots will share the same y-scale. Otherwise, they will have independent scales.
            plot_type ('line', 'scatter'): Defines the type of plot displayed. Default is 'line'.
            figfilename (str): Will save the image, and assign it with file name "figfilename".
        
        Raises:
            AssertError:
                If y2 is provided, y and y2 are of different dimension.
                If ordering is provided, size of ordering is different from total number of sensors.
            Warning:
                'log' option for x_axis_type is buggy, will be fixed soon.
        '''
        import matplotlib.pyplot as plt

        if times is None:
            times = self.time

        # TODO: Fix this
        if x_axis_type == "log":
            warnings.warn("'log' option for x_axis_type is buggy, will be fixed soon")
        # Find where provided time series is within overall series
        start_ind = np.where(self.time == times[0])[0][0]
        end_ind = np.where(self.time == times[-1])[0][0]
        duration = (end_ind - start_ind) / self.fs

        # Only look at y provided within time series
        ydata = y[:, start_ind:end_ind+1]

        fig, ax = plt.subplots(plots_shape[0], plots_shape[1], sharex=sharex, sharey=sharey, figsize=(10, 5/3*self.num_sens), tight_layout=True)

        fig.text(0.5, -0.015, x_label, ha='center')
        fig.text(-0.015, 0.5, y_label, va='center', rotation='vertical')

        if plots_shape[0] == 1:
            ax = np.reshape(ax, (1, plots_shape[1]))
        elif plots_shape[1] == 1:
            ax = np.reshape(ax, (plots_shape[0], 1))
        
        # Create a seperate vertical axis if secondary data source is provided
        if y2 is not None:
            assert y2.shape == y.shape, f"Additional data shape of {y2.shape} does not match primary data shape of {y.shape}."
            # Only look at y2 provided within time series
            y2data = y2[:, start_ind:end_ind+1]

            if y2_label is not None:
                fig.text(1.015, 0.5, y2_label, va='center', rotation=-90)

            ax2 = np.empty(ax.shape, dtype=object)
            for i in range(ax2.shape[0]):
                for j in range(ax2.shape[1]):
                    ax2[i][j] = ax[i][j].twinx()
        
        if ordering is None: 
            ordering = np.arange(ydata.shape[0])
        
        assert len(ordering) == ydata.shape[0] and max(ordering) == ydata.shape[0]-1, f"Make sure ordering contains correct sensor numbers from 0 to {y.shape[0]-1}."
        for count, ind_sens in enumerate(ordering):
            # Determining where sensors are on plot
            i = count // ax.shape[1]
            j = count % ax.shape[1]

            if x.shape[0] == ydata.shape[0]:
                x_select = x[ind_sens]
            else:
                x_select = times
        
            # Plotting graphs
            if plot_type == 'line':
                ax[i][j].plot(x_select, ydata[ind_sens], color=color, linewidth=0.5)
                if y2 is not None:
                    # Plot second data source
                    ax2[i][j].plot(x_select, y2data[ind_sens], color=color2, linewidth=0.5)

            elif plot_type == 'scatter':
                ax[i][j].scatter(x_select, ydata[ind_sens], color=color, s=2)
                if y2 is not None:
                    # Plot second data source
                    ax2[i][j].scatter(x_select, y2data[ind_sens], color=color2, s=2)
            
            # Additional plot features
            ax[i][j].set_xscale(x_axis_type)
            ax[i][j].set_title(f"Sensor {ind_sens + 1}")
            ax[i][j].set_xlim((np.min(times) - duration*0.01, np.max(times) + duration*0.01))

        if figfilename is not None:
            fig.savefig(figfilename,transparent=True)

    def plot_eruption_PT(self, sharex:bool=True, sharey:bool=True, xlim:tuple=(98, 104), ylim:tuple=(0.95, 1.225), times:np.ndarray=None, title:str=None, 
                         show_phase_boundaries:bool=True, ordering:np.ndarray=np.arange(0, 6, dtype=int), cmap:str='summer', savefig:bool=False):
        '''
        Outputs a Pressure-Temperature plot of an eruption. Utilizes the matplotlib module.

        Parameters:
            sharex (bool): Option to use same temperature axis for all sensor plots. Default set to True.
            sharey (bool): Option to use same pressure axis for all sensor plots. Default set to True.
            xlim (tuple): Select temperature range in degrees Celsius. Default is (98, 104).
            ylim (tuple): Select pressure range in bar. Default is (0.95, 1.225).
            times (ndarray): Optional time series splice for the data. Use @method getTimeRange helper to get splice. Default set to self.time.
            title (str): Title for figure, and file name to be used to save the figure if savefig is True. Default is None.
            show_phase_boundaries (bool): Option to show the phase boundaries produced by the Clapeyron equation. Default is False.
            ordering (ndarray): Ordering of sensors to be shown. Default is sensors ordered in ascending order.
            cmap (str): String Representation of color maps from the matplotlib.cm library. Defualt is 'summer'.
            savefig (bool): Option to save PT plot. If True, file name will be the modified title name.
        '''
        import warnings
        from urllib.error import URLError
        import matplotlib.pyplot as plt
        import pandas as pd

        if show_phase_boundaries:
            try: # Load in coefficients for phase boundary
                n = pd.read_csv("iapws-if97-region4.csv")["ni"].to_numpy()
            except FileNotFoundError:
                try:
                    n = pd.read_csv("https://raw.githubusercontent.com/akyaparla/Binary-File-Unpack/main/iapws-if97-region4.csv")["ni"].to_numpy()
                except URLError:
                    warnings.warn("'iapws-if97-region4.csv' file not found, ignore phase boundaries")
                    show_phase_boundaries = False

        ordering = np.array(ordering)
        if times is None: 
            times = self.time
        start_ind = np.where(self.time == times[0])[0][0]
        end_ind = np.where(self.time == times[-1])[0][0]
        P = self.P[:, start_ind:end_ind+1]
        T = self.T[:, start_ind:end_ind+1]

        num = ordering.shape[0]
        fig, ax = plt.subplots(1, num, sharex=sharex, sharey=sharey, figsize=(10/3*num, 10), constrained_layout=True)

        fig.text(0.5, 0.12, r"Temperature ($^{\circ}C$)", ha='center')
        fig.text(-0.015, 0.5, "Pressure (bar)", va='center', rotation='vertical')
        fig.text(0.5, 1.015, title, ha='center')
        
        for count, ind_sens in enumerate(ordering):
            if not sharex:
                minT = np.min(T[ind_sens])
                maxT = np.max(T[ind_sens])
                durationX = maxT - minT
                xlim = (minT - 0.01*durationX, maxT + 0.01*durationX)
            else:
                minT, maxT = xlim

            if not sharey:
                minP = np.min(P[ind_sens])
                maxP = np.max(P[ind_sens])
                durationY = maxP - minP
                ylim = (minP - 0.01*durationY, maxP + 0.01*durationY)
            else:
                minP, maxP = ylim
            # Colormap by time
            data = ax[count].scatter(T[ind_sens], P[ind_sens], c=times, cmap=cmap, s=2)

            P100 = 1.0141797792131013
            Pcurve = np.zeros(50) + P100
            if show_phase_boundaries:          
                Trange = np.linspace(minT, maxT) + 273.15
                theta = Trange + n[8] / (Trange - n[9])
                A =      theta**2 + n[0]*theta + n[1]
                B = n[2]*theta**2 + n[3]*theta + n[4]
                C = n[5]*theta**2 + n[6]*theta + n[7]
                Pcurve = 10 * (((2*C) / (-B + np.sqrt(B**2 - 4*A*C))))**4
                ax[count].plot(Trange - 273.15, Pcurve, 'r--', label="Phase Boundary")
                ax[count].legend(loc="upper right")
            
            # 100 C and 1 bar lines
            if maxT > 100 and minT < 100:
                ax[count].plot([100, 100], [min(minP, np.min(Pcurve)), max(maxP, np.max(Pcurve))], 'k--', alpha=0.5)
            if max(maxP, np.max(Pcurve)) > P100 and min(minP, np.min(Pcurve)) < P100:
                ax[count].plot([min(minT, 100), maxT], [P100, P100], 'k--', alpha=0.5)
                    
            # Additional plot features
            ax[count].set_xlim(xlim)
            ax[count].set_ylim(ylim)
            ax[count].invert_yaxis()
            ax[count].set_title(f"Sensor {ind_sens + 1}")

        # Colorbar
        fig.colorbar(data, ax=ax, shrink=0.9, label="Time (s)", location='bottom', pad=0.05)

        if savefig:
            fig.savefig(title.lower().replace('.', '').replace(' ', '_').replace('\n', '_').replace('(', '-').replace(')', '-')+'.png', bbox_inches="tight")

    def getTimeRange(self, start:float=0, end:float=None) -> np.ndarray:
        '''
        Returns a time series of the data. Intended for use as a time series input for other relavent methods.

        Parameters:
            start (float): The starting time for the slice. Default is 0
            end (float): The end time for the slice. Default is end of time series

        Returns:
            (ndarray): The resulting time series slice for the provided start and end times.

        Raises:
            ValueError: 
                Provided times out of bounds, arguments must be between 0 and {round(len(self.time) / self.fs, 2)}.
                Start time must be less than end time.
        
        '''
        if end is None:
            end = self.time[-1]
        start_index = int((start-self.start_splice) * self.fs)
        end_index = int((end-self.start_splice) * self.fs)
        if start > end:
            raise ValueError(f"Start time must be less than end time.")

        if start_index < 0 or end_index-1 > len(self.time):
            raise ValueError(f"Provided times out of bounds, arguments must be between 0 and {round(len(self.time) / self.fs, 2)}.") 
                
        return self.time[start_index:end_index]
          
    def getDtype(self, typeStr:str, numBytes:int, numObjects:int=1):
        '''
        Method to read the bytes of a file as a given data type. Main purpose is to be used in __init__ method.

        Parameters:
            typeStr (str): String containing data types of requested objects. See https://docs.python.org/3/library/struct.html#format-characters
            numBytes (int): The number of bytes for all data types indicated in @typeStr
            numObjects (int): The number of the inputted typeStr to read. Default is 1.

        Returns:
            Either an object of type np.ndarray or the indicated data type (param type) containing said data types read from the file.
        '''
        import struct
        tmp = self.offset
        self.offset += numBytes*numObjects
        # Total number of requested objects
        totNumObjects = int(numObjects*len(typeStr))
        if totNumObjects == 1:
            return struct.unpack(typeStr, self.fileContent[tmp:self.offset])[0]
        return np.array(struct.unpack(typeStr*numObjects, self.fileContent[tmp:self.offset]))

    def endOfFile(self) -> bool:
        '''
        Checks whether the end of file has been reached. Main purpose is to be used in __init__ method.

        Returns:
            (bool) True if end of file is reached; otherwise False.
        '''
        return self.offset == len(self.fileContent)