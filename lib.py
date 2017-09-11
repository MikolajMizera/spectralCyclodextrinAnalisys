from os import makedirs
from os.path import isdir, join, split
from glob import glob
import warnings

from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

from tpot import TPOTClassifier
#from neat import NEATClassifier

try:
    import plotly.offline as offline
    import plotly.graph_objs as go
    PLOTLY=True
except ImportError as e:
    print('Cannot import plotly, only matplotlib plots will be avaiable')
    PLOTLY = False
    
    
class app:
    
    shortcuts={
            'ca': r'cefuroxime_axetil',
            'ct': r'cefetamet_pivoxil',
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
                 method='tpot',
                 pop_size=24,
                 generations = 10,
                 processes=1,
                 use_scoop=True,
                 verbose=0):
        """
        Initialization method of class app.
        
        Parameters
        ----------
            ftir_dir : str
                A name of directory containing transmission FTIR spectra in 
                format "wavelength<<delimiter>>intensity\n".
            dsc_file : str
                A path to file contataing values indicating presence of complex
                in given sample.
            atr_dir : str
                Optional, a name of directory containg ATR spectra in format
                "wavelength<<delimiter>>intensity\n".
                default = None
            output_dir : str
                A name of directory to save output files (plots, results pickles).
                default = "outs"
            method : str
                A name of one of calssifiers implemented. 
                Possible choices: tpot/neat.
                default = "tpot"
            pop_size : int
                Size of population used for evolutionary optimization of model.
                default = 240
            generations : int
                Number of generations (iterations) of evolutionary optimization.
                defult = 100
            processes : int
                Number of parallel process used for optimization.
                default = 1 (serial execution)
            use_scoop : bool
                Parameter indicating wheather scoop library should be used as 
                parallel backend.
                default = True
            verbose : int
                A level of verbosity.
                0 - only error messages are reported
                1 - only final scores are reported + LEVEL 0
                2 - NEAT evolution is reported + LEVEL 1
                3 - system information + LEVEL 2
                default = 1
        
        """
        
        self.ftir_dir = ftir_dir
        self.dsc_file = dsc_file
        self.atr_dir = atr_dir
        self.output_dir = output_dir
        self.method=method
        self.pop_size = pop_size
        self.generations = generations
        self.processes = processes
        self.use_scoop = use_scoop
        self.verbose = verbose
        self.spectra={}
        self.thermo={}
        
        if not isdir(output_dir):
            makedirs(output_dir)
            
        if self.verbose>1:
            print('Initializing application with parameters:')
            print('\tDirectories:')
            print('\t\tFT-IR spectra: /%s'%(self.ftir_dir))
            if self.atr_dir:
                print('\t\tATR spectra: /%s'%(self.atr_dir))
            print('\t\tPath to DSC results: /%s'%(self.dsc_file))
            print('\t\tOutput folder: /%s'%(self.output_dir))
            print('\tOptimization parameters:')
            print('\t\tMethod: %s'%(self.method))
            print('\t\tPopulation size: %s'%(self.pop_size))
            print('\t\tGenerations: %s'%(self.generations))
            if self.processes > 1:
                print('\t\tParallel processes: %s'%(self.processes))
                if self.use_scoop:
                   print('\t\tScoop backend will be used for parallel processing') 
            
    
    def run(self, method, train_ratio):
        
        self.parse_spectra()
        
        if method=='tpot':
            self.model = TPOTClassifier(generations=self.generations, 
                                        population_size=self.pop_size,
                                        mutation_rate=0.9,
                                        crossover_rate=0.1,
                                        scoring='accuracy', cv=LeaveOneOut(),
                                        subsample=1.0, n_jobs=1,
                                        max_eval_time_mins=5,
                                        random_state=None, verbosity=2,
                                        disable_update_check=True)
        elif method=='neat':
            raise NotImplemented('NEAT backend is not implemented.')
        #not implemented
            self.model = NEATClassifier(generations=self.generations, 
                                        population_size=self.pop_size,
                                        scoring='accuracy', cv=5,
                                        subsample=1.0, n_jobs=1,
                                        random_state=None,verbosity=2)
#        self.plot_spectra('summary', output_folder='plain')
        self.__pool(5)
#        self.plot_spectra('summary', output_folder='pooled')
        
        X, X_test, y, y_test = self.__create_dataset(train_ratio)
        if self.verbose>1:
            print('Starting model fitting...')
        self.predictions = cross_val_predict(self.model, X, y, cv=LeaveOneOut()) 
        return self.predictions
        
    def parse_thermo(self):
        if self.verbose>1:
            print('Parsing DSC data...')
        self.thermo={}
        with open(self.dsc_file, 'r') as f:
            for fl in f.readlines():
                system = fl.split(' ')[0]
                value = float(fl.split(' ')[1])
                self.thermo[system]=value
    
    def parse_spectra(self):
        """
        
        Parses spectra located in directories specified during initialization.
        
        """
        
        self.spectra={}
        for spectra_dir in [self.ftir_dir, self.atr_dir]:
            if spectra_dir==None:
                continue
            #Get folders containing spectral data of all systems
            systems_dirs=[d for d in glob(join(spectra_dir,'*')) if isdir(d)]
            if not join(spectra_dir,'apis') in systems_dirs:
                raise RuntimeError('There are no APIs spectra in folder %s.'
                                       %spectra_dir)
                continue
            if not join(spectra_dir,'cyclodextrins') in systems_dirs:
                raise RuntimeError('There are no cyclodextrins spectra in folder %s.'
                                       %spectra_dir)
                continue
            
            spectrum_type=split(spectra_dir)[-1]
            if self.verbose > 1:
                print('Parsing spectra of type %s...'%spectrum_type)
            self.spectra[spectrum_type]={}
            for system_dir in systems_dirs:
                system=split(system_dir)[-1]
                self.spectra[spectrum_type][system]={} 
                spectra_files=glob(join(system_dir,'*'))                   
                if not len(spectra_files):
                    if self.verbose > 1:
                        print('The directory %s contains no spectra.'%system_dir)
                        continue
                for s_file in spectra_files:
                    #Parse spectral data in format x<delimiter>y and store in dictionary
                    spectrum_name = split(s_file)[-1].split('.')[0]
                    parsed=self.__parse_spectrum(s_file)
                    self.spectra[spectrum_type][system][spectrum_name]=parsed
        if not len(self.spectra):
            raise RuntimeError('Could not parse any spectra.')
        if self.verbose>1:
            print('Finished parsing spectra...')
        self.__interpolate_spectra()
                
    def plot_spectra(self, style='single', output_folder='plots', 
                     engine='matplotlib', spacing=0.5):
        """
        The method plots all parsed spectra in publication-ready format.
        
        Parameters
        ----------
            style : str
                Options: "single"/"summary"
                The single style plots every spectra on spearate plot.
                The summary style plots all spectra of given API-CD system on 
                single plot with common x axis.
                default = "single"
            output_folder : str 
                Name of directory the plots should by saved into.
                default = "plots"
            engine : str
                Options: "matplotlib"/"plotly"
                Plotting engine to use.
                default = "matplotlib"
            spacing : float 
                A ratio of maximum intensity used for spacing calculation.
                space between spectra = max(spectrum) * spacing
                Relevant only in "summary" plotting mode.
                default = 0.5
                            
        """
        
        assert style=='single' or style=='summary'
        assert engine=='matplotlib' or engine=='plotly'
        
        if engine=='plotly' and not PLOTLY:
            warnings.warn('Plotly engine was specified while is not avaiable')
            warnings.warn('Falling back to matplotlib')
            engine='matplotlib'
            
        #Check if there are parsed spectra already, if not - try to parse
        if not len(self.spectra):
            self.parse_spectra()
        
        #Create plot subfolder of not exists                
        if not isdir(join(self.output_dir, output_folder)):
            makedirs(join(self.output_dir, output_folder))
            
        for spectra_key, systems in self.spectra.items():
            systems = [k.split('_') for k in systems.keys() if len(k.split('_'))>1]

            for system in systems:
                
                if self.verbose > 1:
                    print('Plotting %s spectra %s'%(spectra_key,'-'.join(system)))
                
                #variables required for adding spacing
                prev_spectrum_max = 0
                curr_spacing = 0
                
                #Gathering parameters and settings for plotting to zip and iterate
                keys1=['apis', 'cyclodextrins', '_'.join(system), '_'.join(system)]
                keys2=[system[0], system[1], 'mixture', 'complex']
                colors=['black', 'red', 'violet', 'green']
                labels = [system[0], system[1], 
                          'Mixture %s - %s'%(system[0],system[1]),
                          'Complex %s - %s'%(system[0],system[1])]
                plots_params=zip(keys1,keys2, colors, labels)
                
                if style=='summary':
                    self.__setup_figure(spectra_key, system)
                
                for key1, key2, color, label in plots_params:
                    
                    if label in list(self.shortcuts.keys()):
                        label=self.shortcuts[label]
                    
                    curr_spacing += prev_spectrum_max * spacing + prev_spectrum_max
                    
                    if style=='single':
                        self.__setup_figure(spectra_key, system)
                    
                    spectrum=self.spectra[spectra_key][key1][key2]
                    x=spectrum[:,0]
                    y=spectrum[:,1].copy()
                    prev_spectrum_max=np.max(y)
                    y+=curr_spacing
                    plt.plot(x,y, 
                             color=color, 
                             linewidth=0.5, 
                             linestyle='-', 
                             label=label)
                    #Add description of each spectrum near spectrum tail
                    if style=='summary':
                        plt.text(np.max(x), 
                                 np.max(y)-0.75*prev_spectrum_max,                                 
                                 label.split(' ')[0].replace('_',' '))
                    
                    if style=='single':
                        plot_fname='%s_%s_%s.png'%(spectra_key, key1, key2)
                        plt.savefig((join(self.output_dir, output_folder,plot_fname)))
                        plt.close()
                
                if style=='summary':
                    plt.tight_layout(rect=[0, 0, 0.8, 1])
                    plot_fname='%s_%s_%s.png'%(spectra_key, system[0], system[1])
                    plt.savefig((join(self.output_dir, output_folder,plot_fname)))  
                    plt.close()
    
    def __setup_figure(self, spectra_key, system):
        plt.figure()
        plt.title('%s spectrum of system %s - %s'%(spectra_key, system[0].upper(), system[1].upper()))
        plt.xlim([400,4200])
        plt.xticks(np.arange(400, 4001, 400))
        plt.yticks([])
        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)
        plt.gca().set_xlabel('Wavelength (cm^-1)')
        plt.gca().set_ylabel('Intenisty/Absorbtion')

    def __pool(self, window):
        for methods in self.spectra.keys():
            for systems in self.spectra[methods].keys():
                for spectrum_k in self.spectra[methods][systems].keys():
                    spectrum=self.spectra[methods][systems][spectrum_k]
                    new_y=self.__apply_pooling(spectrum[:,1], window, method='max')
                    new_x=self.__apply_pooling(spectrum[:,0], window, method='min')
                    self.spectra[methods][systems][spectrum_k]=np.array([new_x, new_y]).T
        if self.verbose>1:
            print('Finished pooling...')

    
    def __create_dataset(self, train_ratio):
        """
        Divides spectra into datasets ready for machine learning.
        
        Parameters
        ----------
            train_ratio : float
                fraction of dataset used for model training
        
        Returns
        -------
            X_train : numpy array
                A set of input vectors for training.
            X_test : numpy array
                A set of input vectors for testing.
            y_train : numpy array
                A set of target values for training.
            y_test : numpy array
                A set of target values for testing.
        """ 
        #Check if there are parsed spectra already, if not - try to parse
        if not len(self.spectra):
            self.parse_spectra()
        if not len(self.thermo):
            self.parse_thermo()

        test_ratio = 1 - train_ratio
        #The dictionary to store concatenated spectra measured with different spectral  methods
        X = {k:[] for k in self.spectra['ftir'].keys() if 
                                             k!='apis' and k!='cyclodextrins'}
        X_comp=X.copy()
        X_mixt=X.copy()
        #Merge spectra of different methods (FTIR, ATR...) for each system
        for method in self.spectra.values():
            for system in X.keys():
                sys_spectra=method[system]
                api_spectrum=method['apis'][system.split('_')[0]][:,1]
                cd_spectrum=method['cyclodextrins'][system.split('_')[1]][:,1]
                mixt_spectrum=sys_spectra['mixture'][:,1]
                complex_spectrum=sys_spectra['complex'][:,1]
                spectra=np.concatenate(
                        [api_spectrum, cd_spectrum, mixt_spectrum, complex_spectrum])
                spectra_comp=np.concatenate(
                        [api_spectrum, cd_spectrum, complex_spectrum])
                spectra_mixt=np.concatenate(
                        [api_spectrum, cd_spectrum, mixt_spectrum])
                X[system]=np.concatenate([X[system], spectra])
                X_comp[system]=np.concatenate([X_comp[system], spectra_comp])
                X_mixt[system]=np.concatenate([X_mixt[system], spectra_mixt])
                
        labels = list(X.keys())
        y = self.thermo
#        X_train, y_train = zip(*[(X[k], y[k]) for k in X.keys()])
        X_train = list(X_comp.values()) + list(X_mixt.values())
        y_train = [1]*len(X_comp) + [0]*len(X_mixt)
        
        X_test = y_test = []
        if train_ratio<1:
            X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                                y_train, 
                                                                train_size=train_ratio,
                                                                test_size=test_ratio)
        if self.verbose>1:
            print('Created dataset of %d training examples and %d test examples'%
                  (len(X_train), len(X_test)))
        return np.array(X_train), X_test,  np.array(y_train), y_test
        
    def __apply_pooling(self, spectrum, window, method='max'):
        """ 
        Calculate rooling window value of a function over data
        
        Parameters
        ----------
            spectrum : iterable
                1D iterable of size n
            window_size : int 
                size of window
            method : string
                a function to apply on each window (max/min/mean)
        
        Returns
        -------
            data : numpy array
                data after pooling of size n/3
        """
        pooling_fcns={'max': max,
                      'min': min,
                      'mean': np.mean}
        data=[]
        pooling_fcn=pooling_fcns[method]
        windows=int(np.ceil(len(spectrum)/window))
        for i in range(0, windows, window):
            offset=min([window, len(spectrum)-i])
            data.append(pooling_fcn(spectrum[i:i+offset]))
        return data

    def __interpolate_spectra(self):
        """
        
        Interpolates spectra so all of them have common wavelength domain (X)
        
        """
        #Check if there are parsed spectra already, if not - parse
        if not len(self.spectra):
            self.parse_spectra()

        min_x, max_x = self.__find_common_bounds()
        new_x = np.arange(min_x, max_x)
        for methods in self.spectra.keys():
            for systems in self.spectra[methods].keys():
                for spectrum_k in self.spectra[methods][systems].keys():
                    spectrum=self.spectra[methods][systems][spectrum_k]
                    f = interpolate.interp1d(spectrum[:,0], spectrum[:,1])
                    new_y=f(new_x)
                    self.spectra[methods][systems][spectrum_k]=np.array([new_x, new_y]).T
        if self.verbose>1:
            print('Finished spectra interpolation to range (%d, %d)...'%(min_x, max_x))
    
    def __find_common_bounds(self):
        #Find maximal common minimum wavelength
        all_spectra = self.__flatten(self.spectra)
        min_x=max([min(x[:,0]) for x in all_spectra])
        #Find minimal common maximum wavelength
        max_x=min([max(x[:,0]) for x in all_spectra])
        return min_x, max_x
    
    def __parse_spectrum(self, file, normalize=True):
        """ 
        
        Loads txt file with spectral data requires spectrum in 
        format [wavelength, intensity]
        
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
    
    def __flatten(self, d):    
        """
        
        Utillity that flattens dict to get list of numpy arrays
        
        """
        res = []  # Result list
        if isinstance(d, dict):
            for key, val in d.items():
                res.extend(self.__flatten(val))
        elif isinstance(d, np.ndarray):
            res = [d]        
        else:
            raise TypeError("Undefined type for flatten: %s"%type(d))
        return res