#!/rugpfs/fs0/cem/store/mreynolds/software/miniconda3/envs/matt_picker4/bin/python
################################################################################
# imports
print('Beginning imports...')
import numpy as np
import xml.etree.ElementTree as ET
import os
print('Imports finished. Beginning script...')
################################################################################
import argparse; import sys
parser = argparse.ArgumentParser('Generate noisy and noiseless 2D projections of randomly oriented and translated MRC files.')
parser.add_argument('--tilt_series', type=str, help='tilt series directory name')
parser.add_argument('--cmm1', type=str, help='cmm file to concatenate')
parser.add_argument('--cmm2', type=str, help='cmm file to concatenate')

################################################################################
################################################################################
def combine_files(filenames, output_filename):
    if not filenames:
        return
    
    # Parse the first file to initialize the root
    tree = ET.parse(filenames[0])
    root = tree.getroot()
    
    # Get the color attributes of the first marker
    first_marker = root[0]
    r = first_marker.attrib["r"]
    g = first_marker.attrib["g"]
    b = first_marker.attrib["b"]
    
    # Starting marker id after the first file
    last_id = int(root[-1].attrib["id"])

    # Process the rest of the files
    for file in filenames[1:]:
        tree_temp = ET.parse(file)
        for marker in tree_temp.getroot():
            # Increment marker id
            last_id += 1
            marker.attrib["id"] = str(last_id)
            
            # Set the color attributes to be the same as the first marker
            marker.attrib["r"] = r
            marker.attrib["g"] = g
            marker.attrib["b"] = b

            root.append(marker)
    output_filename = '_'.join([os.path.basename(f).replace('.cmm', '') for f in filenames]) + '.cmm'
    output_filepath = os.path.join(base_path, output_filename)
    # Write combined XML to the output file
    tree.write(output_filepath)


if __name__ == "__main__":
    args = parser.parse_args()
    base_path = '/rugpfs/fs0/cem/store/mreynolds/fascin_tomos/subtomo_averaging/alpha_values_bin1/bilds/automated_fil_identification_curated/'
    base_path = base_path + args.tilt_series#'ts045/'
    filenames = [base_path+args.cmm1, base_path+args.cmm2]  # Add your filenames here
    output_filename = base_path +'combined.cmm'
    combine_files(filenames, output_filename)
    print(f"Combined {len(filenames)} files into {output_filename}!")