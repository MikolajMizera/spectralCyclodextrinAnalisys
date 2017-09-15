"""

A script used for classifcation of spectra of cyclodextrins  - APIs systems into 
one of two classes: complexed/uncomplexed.

The input data used for building feature vectors constitute absorbtion ftir 
and/or atr data from specified folders.
Example input spectrum:
    file "ca.csv" - contains cefuroxime axetil spectrum
        <wavelength><delimiter><absorbtion><newline>

The target data are read from file and should contain information wheather 
given sample contain complex and should be marked as binary 1. If given sample 
contains phisical mixture or not complexed sample, it should be marked with binary 0.
Example target file:
    file "dsc" - contains target data
        <system><space><0/1><newline>

Naming convention
The porgramm uses following folder naming convention:
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

Basing on "ftir" folder, the porgramm will discover all investigated samples 
and will create internal databse and datasets for machine learning.

"""
import argparse

from lib import app

VERSION = '0.4.0_RC1'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ftir', '--ftir_spectra_dir', 
                    help='path to directory containing absorbtion FTIR spectra of samples',
                    type=str)
    parser.add_argument('-dsc', '--dsc_file', 
                    help='file with dsc results',
                    type=str)
    parser.add_argument('-atr', '--atr_spectra_dir', 
                    help='path to directory containing ATR-FTIR spectra of samples (default = None)',
                    type=str, default=None)
    parser.add_argument('-o', '--output_dir', help='folder for output files (default = out)',
                    type=str, default='outs')
    parser.add_argument('-m', '--optimization_method', 
                    help='model optimization method (tpot/tree) (default = tpot)',
                    type=str, default='tpot')
    parser.add_argument('-p', '--population_size', 
                    help='size of population for evolution (default = 12)',
                    type=int, default=12)
    parser.add_argument('--processes', 
                    help='processes to use in parallel processing (default = 1)', 
                    type=int, default=1)
    parser.add_argument('-l', '--limit',
                    help='upper limit of spectra wavelength, only data below limit will be used (default = None)',
                    type=int, default=None)
    parser.add_argument('--pool',
                    help='window size for pooling spectra for dimensionality reduction (default = None)',
                    type=int, default=None)
    parser.add_argument('--verbose', 
                    help='verbosity level (default = 1)', 
                    type=int, default=1)
    args=parser.parse_args()
    
    ftir_dir = args.ftir_spectra_dir
    dsc_file=args.dsc_file
    atr_dir = args.atr_spectra_dir
    output_dir = args.output_dir
    method = args.optimization_method
    pop_size = args.population_size
    processes = args.processes
    pool = args.pool
    limit = args.limit
    verbose = args.verbose
    
    app_instance = app(ftir_dir, dsc_file, 
                   atr_dir=atr_dir, 
                   output_dir=output_dir,
                   method=method,
                   pop_size=pop_size,
                   processes=processes,
                   limit=limit,
                   pool=pool,
                   verbose=verbose)


    app_instance.run()