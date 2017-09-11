"""
The programm


Naming convention
The porgramm uses following naming convention:
    1. Spectral data should be placed in "ftir" (obligatory) and "atr" (optional)
    2. Subfolders should contain spectra of investigated systems eg. :
        - ftir
            - cyclodextrins         # for FTIR spectra of cyclodextrins
                - acd.csv
            - apis                  # for FTIR spectra of APIs
                - ca.csv
            - ca_acd                # for FTIR spectra of API - CD system
                - complex.csv
                - mixture.tsv
        - atr
            - cyclodextrins         # for ATR spectra of cyclodextrins
                - acd.csv
            - apis                  # for ATR spectra of APIs
                - ca.csv
            - ca_acd                # for ATR spectra of API - CD system
                - complex.csv
                - mixture.tsv
    3. File with DSC results should be in text format <system 0/1 /newline/> eg. "ca_acd 1",
    which translates to "the complex formation in sample ca_acd was confirmed"
Basing on "ftir" folder, the porgramm will discover all investigated samples and 
create internal databse and datasets for machine learning.

"""
import argparse

from lib import app

VERSION = '0.2.0'

parser = argparse.ArgumentParser()
parser.add_argument('-ftir', '--ftir_spectra_dir', 
                    help='path to directory containing FTIR spectra of samples',
                    type=str)
parser.add_argument('-dsc', '--dsc_file', 
                    help='file with dsc results',
                    type=str)
parser.add_argument('-atr', '--atr_spectra_dir', 
                    help='path to directory containing FTIR spectra of samples (default = None)',
                    type=str, default=None)
parser.add_argument('-o', '--output_dir', help='folder for output files (default = out)',
                    type=str, default='outs')
parser.add_argument('-m', '--optimization_method', 
                    help='model optimization method (tpot/neat) (default = tpot)',
                    type=str, default='tpot')
parser.add_argument('-p', '--population_size', 
                    help='size of population for evolution (default = 240)',
                    type=int, default=12)
parser.add_argument('--processes', 
                    help='processes to use in parallel processing (default = 1)', 
                    type=int, default=1)
parser.add_argument('--use_scoop', 
                    help='do you want to use scoop as multiprocessing library? (default = True)', 
                    type=bool, default=True)
parser.add_argument('--verbose', 
                    help='verbosity level (default = 1)', 
                    type=int, default=0)
args=parser.parse_args()

ftir_dir = args.ftir_spectra_dir
dsc_file=args.dsc_file
atr_dir = args.atr_spectra_dir
output_dir = args.output_dir
method = args.optimization_method
pop_size = args.population_size
processes = args.processes
use_scoop = args.use_scoop
verbose = args.verbose

app_instance = app(ftir_dir, dsc_file, 
                   atr_dir=atr_dir, 
                   output_dir=output_dir,
                   method=method,
                   pop_size=pop_size,
                   processes=processes,
                   use_scoop=use_scoop,
                   verbose=verbose)

app_instance.run('tpot', 1.0)