Wafer Bow and Warp Analysis Tool

Overview
The Wafer Bow and Warp Analysis Tool is a comprehensive Python-based application designed to analyze wafer data, compute bow and warp metrics, and generate detailed heatmaps. The tool provides a user-friendly GUI interface for selecting input files and folders, configuring analysis parameters, and visualizing results.

Features
Lot and Wafer Identification: Automatically extracts lot and wafer information from directory structures and file names.
Plane Fitting: Fits a plane to the provided points and values for tilt correction.
Bow and Warp Calculation: Computes bow and warp metrics with optional outlier filtering and radius-based tilt correction.
Heatmap Generation: Generates 2D heatmaps of the wafer data using Plotly.
Data Analysis: Reads and concatenates data from multiple .txt files, performs statistical analysis, and saves results.

Advanced Features:
Outlier Filtering with Z-Score Threshold
Radius-Based Tilt Correction
Actual Center Computation

Dependencies
The tool requires the following Python libraries:

os
re
pandas
tkinter
scipy
numpy
logging
openpyxl
plotly
concurrent.futures

You can install these dependencies using pip:

bash
pip install pandas tkinter scipy numpy logging openpyxl plotly

Usage
Running the GUI
To run the GUI application, execute the following command:

bash
python Warpspeed_v2.04.py

GUI Instructions
Add Folder: Select folders containing wafer data files.
Add Text File: Select individual text files containing wafer data.
Remove Selected: Remove selected folders or files from the list.
Output Folder: Select the folder where results will be saved.
Generate 2D Heatmaps: Enable or disable heatmap generation.
Interpolation Method: Choose the interpolation method for heatmaps (cubic or linear).
Advanced Features: Configure additional analysis options:
Enable Outlier Filtering
Set Z-Score Threshold
Enable Radius-Based Tilt Correction
Set Radius Value
Enable Actual Center Computation
Thresholds: Set USL and Control Limit values for bow and warp.
File Names: Specify base names for output files.
Run Analysis: Start the analysis process.
Exit: Close the application.

Output
The tool generates the following output files in the specified output folder:

Combined X, Y, Z data in CSV format.
Bow and Warp analysis results in Excel format with conditional formatting.
2D heatmaps in HTML format (if enabled).
Logging
The tool uses Python's built-in logging module to provide detailed logs of the analysis process. Logs are displayed in the console for easy debugging and monitoring.

Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.

Contact
For any questions or suggestions, please contact Timothy.Lund@skywatertechnology.com
