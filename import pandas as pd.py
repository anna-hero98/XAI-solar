import pandas as pd
import h5py
import numpy as np
import os

# h5 example file
h5_file_path = 'SOLETE_Pombo_60min.h5'


def h5_to_excel_or_csv(h5_file, output_file=None, output_format='csv'):
    """
    Reads an HDF5 file, displays its metadata, and converts it to an Excel or CSV file.
    Parameters:
    h5_file (str): Path to the HDF5 file.
    output_file (str): Path to the output file (without extension). If None, the same name as the HDF5 file will be used.
    output_format (str): 'csv' or 'excel' for the output format.
    """

    # Display metadata
  

    # Read the data
    df = pd.read_hdf(h5_file, key="DATA")

    # Determine output file name
    if output_file is None:
        output_file = os.path.splitext(h5_file)[0]

    # Convert to CSV or Excel
    if output_format.lower() == 'csv':
        df.to_csv(f"{output_file}.csv", index=False)
        print(f"Data converted to CSV: {output_file}.csv")
    elif output_format.lower() == 'excel':
        df.to_excel(f"{output_file}.xlsx", index=False)
        print(f"Data converted to Excel: {output_file}.xlsx")
    else:
        raise ValueError("Unsupported output format. Choose 'csv' or 'excel'.")




if __name__ == '__main__':
    h5_to_excel_or_csv(h5_file_path, output_format='excel')
    pass