from os import makedirs
from os.path import isdir, join, split
from glob import glob
import warnings

from matplotlib import pyplot as plt
import numpy as np

try:
    import plotly.offline as offline
    import plotly.graph_objs as go
    PLOTLY=True
except ImportError as e:
    print('Cannot import plotly, only matplotlib plots will be avaiable')
    PLOTLY = False
    
    
"""
Verbosity levels:
    0 - only error messages are reported
    1 - only final scores are reported + LEVEL 0
    2 - NEAT evolution is reported + LEVEL 1
    3 - system information + LEVEL 2
"""

class app:
    
    shortcuts={
            'ca': r'cefuroxime axetil',
            'ct': r'cefetamet pivoxil',
            'pa': r'pivampicillin',
            'acd': r'$\alpha$-cyclodextrin',
            'bcd': r'$\beta$-cyclodextrin',
            'gcd': r'$\gamma$-cyclodextrin',
            'mbcd': r'methyl-$\beta$-cyclodextrin',
            'hpacd': r'hp-$\alpha$-cyclodextrin',
            'hpbcd': r'hp-$\beta$-cyclodextrin',
            'hpgcd': r'hp-$\gamma$-cyclodextrin',}
    
    def __init__(self, 
                 ftir_dir, 
                 dsc_file, 
                 atr_dir=None,
                 output_dir='outs', 
                 pop_size=240,
                 processes=1,
                 use_scoop=True,
                 verbose=0):
        self.ftir_dir = ftir_dir
        self.dsc_file = dsc_file
        self.atr_dir = atr_dir
        self.output_dir = output_dir
        self.pop_size = pop_size
        self.processes = processes
        self.use_scoop = use_scoop
        self.verbose = verbose
        self.spectra={}
        if not isdir(output_dir):
            makedirs(output_dir)
        if verbose>2:
            print('Result files will be saved in folder %s'%output_dir)
    
    def parse_spectra(self):
        spectra={}
        for spectra_dir in [self.ftir_dir, self.atr_dir]:
            if spectra_dir==None:
                continue
            #Get folders with systems spectral data
            systems_dirs=[d for d in glob(join(spectra_dir,'*')) if isdir(d)]
            if not join(spectra_dir,'apis') in systems_dirs:
                if self.verbose > 2:
                    print('There are no APIs spectra in folder %s.'%spectra_dir)
                continue
            if not join(spectra_dir,'cyclodextrins') in systems_dirs:
                if self.verbose > 2:
                    print('There are no CDs spectra in folder %s.'%spectra_dir)
                continue
            spectrum_type=spectra_dir
            spectra[spectrum_type]={}
            for system_dir in systems_dirs:
                system=split(system_dir)[-1]
                spectra[spectrum_type][system]={} 
                spectra_files=glob(join(system_dir,'*'))                   
                if not len(spectra_files):
                    if self.verbose > 2:
                        print('The directory %s contains no spectra.'%system_dir)
                        continue
                for s_file in spectra_files:
                    #Parse spectral data in format x<delimiter>y and store in dictionary
                    spectrum_name = split(s_file)[-1].split('.')[0]
                    spectra[spectrum_type][system][spectrum_name]=self.__parse_spectrum(s_file)
        if not len(spectra):
            raise RuntimeError('Could not parse any spectra.')
        return spectra
    
    def plot_spectra(self, style='single', output_folder='plots', 
                     engine='matplotlib', spacing=0.5):
        """
        The method plots all spectra in publication-ready format.
        arguments:
            style= 'single' (default) or 'summary' 
                The single style plots every spectra on spearate plot.
                The summary style plots all spectra of given system 
                (API/CD/mixture/complex) on single plot with common x axis.
            output_folder= 'plots' (default)
                Name of directory the plots should by saved into
            engine = 'matplotlib' (default) ot ;plotly'
                Which plotting engine use to plot spectra
            spacing= 0.5 (default)
                Spacing between plots if plotting in summary mode equals
                max(spectrum) * spacing
        """
        assert style=='single' or style=='summary'
        assert engine=='matplotlib' or engine=='plotly'
        
        if engine=='plotly' and not PLOTLY:
            warnings.warn('Plotly engine was specified while is not avaiable')
            warnings.warn('Falling back to matplotlib')
            engine='matplotlib'
        #Check if there are parsed spectra already, if not - try to parse
        if not len(self.spectra):
#            try:
            self.spectra=self.parse_spectra()
#            except Exception as e:
#                print(e)
#                return
        for spectra_key, systems in self.spectra.items():
            if self.verbose > 2:
                print('Plotting %s spectra'%spectra_key)
            systems = [k.split('_') for k in systems.keys() if len(k.split('_'))>1]

            for system in systems:
                #if plotting in summary style - add spacing between spectra
                #baisng on previous spectrum maximum and spacing argument
                prev_spectrum_max = 0
                curr_spacing = 0
                if self.verbose > 2:
                    print('Plotting %s spectra of system/API/CD %s'%
                          ('_'.join(system),system[0],system[1]))
                #Parameters for plots to zip and iterate
                keys1=['apis', 'cyclodextrins', '_'.join(system), '_'.join(system)]
                keys2=[system[0], system[1], 'mixture', 'complex']
                colors=['black', 'red', 'violet', 'green']
                labels = [system[0], system[1], 
                          'Mixture %s - %s'%(system[0],system[1]),
                          'Complex %s - %s'%(system[0],system[1])]
                plots_params=zip(keys1,keys2, colors, labels)
                if style=='summary':
                        plt.figure()
                        plt.title('%s spectrum of system %s - %s'%(spectra_key, system[0].upper(), system[1].upper()))
                        plt.xlim([400,4200])
                        plt.xticks(np.arange(400, 4001, 400))
                        plt.yticks([])
                        for spine in ['top', 'right']:
                            plt.gca().spines[spine].set_visible(False)
                        plt.gca().set_xlabel('Wavelength (cm^-1)')
                        plt.gca().set_ylabel('Intenisty/Absorbtion')
                for key1, key2, color, label in plots_params:
                    if label in list(self.shortcuts.keys()):
                        label=self.shortcuts[label]
                    curr_spacing += prev_spectrum_max * spacing + prev_spectrum_max
                    if style=='single':
                        plt.figure()
                        plt.title('%s spectrum of system %s - %s'%(spectra_key, key1.upper(), key2.upper()))
                    spectrum=self.spectra[spectra_key][key1][key2]
                    x=spectrum[:,0]
                    y=spectrum[:,1]
                    prev_spectrum_max=np.max(y)
                    y+=curr_spacing
                    plt.plot(x,y, 
                             color=color, 
                             linewidth=0.5, 
                             linestyle='-', 
                             label=label)
                    if style=='summary':
                        plt.text(np.max(x), 
                                 np.max(y)-0.75*prev_spectrum_max,                                 
                                 label.split(' ')[0])
                    if style=='single':
                        plot_fname='%s_%s_%s.png'%(spectra_key, key1, key2)
                        plt.savefig((join(output_folder,plot_fname)))
                        plt.close()
                if style=='summary':
                    plt.tight_layout(rect=[0, 0, 0.8, 1])
                    plot_fname='%s_%s_%s.png'%(spectra_key, system[0], system[1])
                    plt.savefig((join(output_folder,plot_fname)))  
                    plt.close()
                
                
        
    
    def __parse_spectrum(self, file, normalize=True):
        """ Loads txt file with spectral data
            requires sepectrum in format [x, y]
        """
        delimiters=[None, ' ',',','\t']
        spectrum=[]
        for d in delimiters:
            try:
                spectrum=np.loadtxt(file,delimiter=d)
                if len(spectrum):
                    continue
            except:
                if self.verbose>2:
                    print('Parsing file %s failed, trying different delimiter'%file)
                continue
        if not len(spectrum):
            raise RuntimeError('Parsing file %s failed'%file)
            
        if normalize:
            spectrum=self.__normalize_spectrum(spectrum)
        return spectrum 
    
    def __normalize_spectrum(self, spectrum):
        spectrum[:,1]=(spectrum[:,1]-np.min(spectrum[:,1],axis=0))/(np.max(spectrum[:,1],axis=0)-np.min(spectrum[:,1],axis=0))
        return spectrum