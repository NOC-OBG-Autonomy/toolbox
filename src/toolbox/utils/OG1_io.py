"""Contains functions related to the import and export of OG1 data"""

import xarray as xr


class OG1DataObject:
    """OG1 Data Object"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.metadata = None

    def load(self):
        """Load the data from the file path into xarray dataset"""
        self.data = xr.open_dataset(self.file_path, engine="h5netcdf")
        self.metadata = self.data.attrs

    def view_metadata(self):
        """Prints the metadata to the console in a human-readable format"""
        for key, value in self.metadata.items():
            print(f"{key}: {value}")

    def view_data(self):
        """Print the data using the built-in xarray print function"""
        print(self.data)
