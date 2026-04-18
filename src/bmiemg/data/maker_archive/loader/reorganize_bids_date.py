"""This module helps when our archive bids foulder system has date format,
but proper bids goes directly to subejct. This will colapse the downlaoded
dataset into subjects (not dates)"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os

from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
def reorganize_bids_of_date():
    # 3.1 Create a folder before data_dir, temp

    # 3.2 Get a list of child path (hopefully the subjects path)

    # 3.3 Make a list of pairs of (path, subject)

    # 3.4 Creates on temp unique folder per subject

    # 3.5 Creates on temp/subject empty ses folders per nr of datapoints

    # 3.6 Copies data to the earlies empty ses folder

    # 3.6.A Makes sure that temp and root have same size

    # 3.7. Delete the root folder

    # 3.8. Rename the temp folder with the name of the root folder
    pass
