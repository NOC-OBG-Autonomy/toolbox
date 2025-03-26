"""Base class for all toolbox classes."""

from utils.OG1_io import OG1DataObject


if __name__ == "__main__":
    OG1_example_loc = "../examples/data/OG1/ALR_4_649_R.nc"
    OG1 = OG1DataObject(OG1_example_loc)
    # load
    OG1.load()
    # view data
    OG1.view_data()
