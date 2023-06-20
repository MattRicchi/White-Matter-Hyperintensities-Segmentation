#!/usr/bin/env python3
'''
This script is for plotting flair, ground truth and result images

Author: Mattia Ricchi
Date: May 2023
'''

import sys
import getopt
from General_Functions.plot_functions import plot_images

def main(argv):
    patient_number = None
    slice_number = None

    try:
        opts, args = getopt.getopt(argv, "hp:s:", ["patient=", "slice="])
    except getopt.GetoptError:
        print('plot_images.py -p <patient_number> -s <slice_number>')
        sys.exit(2)

    # Parse the command-line options and arguments
    for opt, arg in opts:
        if opt == '-h':
            # Display help message and exit
            print('plot_images.py -p <patient_number> -s <slice_number>')
            sys.exit()
        elif opt in ("-p", "--patient"):
            # Store the patient number
            patient_number = arg
        elif opt in ("-s", "--slice"):
            # Store the slice number
            slice_number = arg

    # Check if both patient number and slice number are provided
    if patient_number is None or slice_number is None:
        print('Please provide both patient number and slice number.')
        sys.exit(2)

    # Call the plot_images function with the provided patient and slice numbers
    plot_images(patient_number, slice_number)

if __name__ == "__main__":
    # Call the main function with the command-line arguments, excluding the script name
    main(sys.argv[1:])
