# Binary-File-Unpack
A Python class to unpack and read binary files intended for LDEO Geyser Project.
Files in repository:
  * *BinaryFileUnpack.py* Contains the Python module for the <code>BinaryFileUnpack</code> class.
  * *HotWaterCycles_heaters_on_steam_in-20221118-20-09-42.bin*: A binary file containing test data.
  * *example.ipynb* A jupyter notebook that tests the <code>BinaryFileUnpack</code> class using the test data.
  * *freq-analysis_cold.ipynb* A jupyter notebook that reads through a given directory containing cold water oscillation data and writes the parameters of the least-squares regression, as well as their standard deviations, to a .csv file. Also includes the X, Y, start time, and end time of the tests.
   * *freq-analysis_hot.ipynb* A jupyter notebook that reads through a given directory containing hot water oscillation data and writes the X, Y, and angular frequencies of each trial along with their standard deviations, to a .csv files. Due to inconsistencies in the oscillation patterms, a spectral analysis is used to extract frequencies, leading to higher error. 
   * *calibration_data-03-31.csv* File containing the calibration data used for getting y level from sensor pressure. The Y* column is the level in the conduit measured from the top of the tank, and the raw Pressure readings indicate the pressure that the respective sensor recorded for the Y* level.
   * *calibration_data_constrict.csv* Same as calibration_data-03-31.csv, but for the constricted conduit.
   * *iapws-if97-region4.csv* Parameters that specify region 4 (saturation curve) on the IAPWS-97 standard. Taken from http://twt.mpei.ac.ru/mcs/worksheets/iapws/IAPWS-IF97-Region4.xmcd
