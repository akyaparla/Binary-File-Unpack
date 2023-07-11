# Binary-File-Unpack
A Python class to unpack and read binary files intended for LDEO Geyser Project.
Files in repository:
  * *BinaryFileUnpack.py* Contains the Python module for the <code>BinaryFileUnpack</code> class.
  * *HotWaterCycles_heaters_on_steam_in-20221118-20-09-42.bin*: A binary file containing test data.
  * *example.ipynb* A jupyter notebook that tests the <code>BinaryFileUnpack</code> class using the test data.
  * *freq-analysis_cold.ipynb* A jupyter notebook that reads through a given directory containing cold water oscillation data and writes the parameters of the least-squares regression, as well as their standard deviations, to a .csv file. Also includes the X, Y, start time, and end time of the tests.
   * *freq-analysis_hot.ipynb* A jupyter notebook that reads through a given directory containing hot water oscillation data and writes the X, Y, and angular frequencies of each trial along with their standard deviations, to a .csv files.
   * *iapws-if97-region4.csv* Parameters that specify region 4 (saturation curve) on the IAPWS-97 standard. Taken from http://twt.mpei.ac.ru/mcs/worksheets/iapws/IAPWS-IF97-Region4.xmcd
