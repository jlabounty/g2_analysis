import os
import g2fit
from g2fit import fitting,helpers,analysis
from g2_analysis import configuration
import datetime
import tomlkit
import gzip 
import pickle
import json
import numpy as np
import pytz
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import hist

methods = {
    0:'t',
    1:'a'
}

def get_date():
    # return datetime.datetime.now(tz='Pacific')
    d = datetime.datetime.now()
    timezone = pytz.timezone("America/Los_Angeles")
    d_aware = timezone.localize(d)
    return d_aware

class PrecessionAnalysis():
    def __init__(self, config_file) -> None:
        self.h = None
        if(type(config_file) is configuration.AnalysisConfig):
            self.config = config_file
            self.config_file = self.config.infile
        else:
            try:
                assert os.path.exists(config_file)
            except:
                raise FileNotFoundError(f'Configuration file not found: {config_file}')
            self.config_file = config_file
            self.config = configuration.AnalysisConfig(self.config_file)

    def apply_pileup_correction(self, force=False, **kwargs):
        '''
            Uses the informatiion in the configuration file to do the pileup correction and return the result
        '''
        if(self.config['pileup_corr']['complete'] and os.path.exists(self.config['pileup_corr']['file'])):
            h = analysis.gm2histogram.load(self.config['pileup_corr']['file'])
        
        else:
            print('Calculating pileup correction from scratch')
            directory = self.config['directory']
            raw_file = self.config['raw']['file']
            assert os.path.exists(raw_file)
            h = analysis.gm2histogram(
                raw_file,
                key =    self.config['pileup_corr']['hist_folder'],
                h2_key = self.config['pileup_corr']['pileup_folder'],
                pileup_scale_factor = self.config['pileup_corr']['scale_factor'],
                rebinfactor = self.config['pileup_corr']['rebin']
            )

            dataset = self.config['dataset']
            name = self.config['pileup_corr']['pileup_name']
            outfile = os.path.join(directory, 'data_pileupCorr', f'results_{dataset}_{name}.pickle')
            h.write(outfile)

            self.config['pileup_corr']['file'] = outfile
            self.config['pileup_corr']['complete'] = True 
            self.config['pileup_corr']['process_time'] = get_date()
            self.config.update()
        
        self.h = h
        return self.h

    def get_fit_params(self, fit='5', method=0):
        these_params = self.config['fitting']['fits'][fit]

        # default to global params
        par_names, par_guesses, par_lim_low, par_lim_high, par_fixed = zip(*these_params['params'])
        
        if(type(method) is int):
            method = methods[method]
        if(method in these_params):
            if(these_params[method]['complete']): # unless the fit was already completed
                par_names, par_guesses, par_lim_low, par_lim_high, par_fixed = zip(*these_params[method]['fitted_params'])
            elif('params' in these_params[method]): # or we have t/a method specific guesses defined
                par_names, par_guesses, par_lim_low, par_lim_high, par_fixed = zip(*these_params[method]['params'])

            
        par_lim_low  = [x if ('None' not in str(x)) else None for x in par_lim_low  ]
        par_lim_high = [x if ('None' not in str(x)) else None for x in par_lim_high ]

        di = {
            "names":par_names, 
            "guess":par_guesses, 
            "limlow":par_lim_low, 
            "limhigh":par_lim_high, 
            "fixed":par_fixed
        }

        if('inherit_params' in these_params and these_params['inherit_params'] and not these_params['complete']):
            additional_params = self.get_fit_params(str(these_params['inherit_from']), method)
            for x in di:
                di[x] = additional_params[x] + di[x]

        return di

    def prepare_fit_function(self, fit='5', method=0):
        this_config = self.config['fitting']['fits'][fit]
        # print(this_config)
        fit_params = self.get_fit_params(fit, method)
        # print(fit_params)
        if(this_config['complete'] or ('function_file' in this_config)):
            print('Loading from fit file!') #TODO: implement file save/load
            outfile = this_config['function_file']
            this_fit = analysis.fitting.FullOmegaAFit.load(outfile)
            # this_fit.re_init()
            assert set(fit_params['names']) == set(this_fit.params), f'ERROR: the params in the loaded file do not match! {fit_params} vs. {this_fit.params}'
        else:
            this_fit = analysis.fitting.FullOmegaAFit(
                self.config['blinding_phrase'],
                fit_params['names'],
                'python',
                loss_spectrum=None #TODO, implement loss spectrum
            )
            this_hash = hash(json.dumps(this_config, sort_keys=True))
            outfile = os.path.join(self.config['directory'], 'fits', f'fit_function_{fit}_{this_hash}.pickle')
            this_fit.save(outfile)
            self.config['fitting']['fits'][fit]['function_file'] = outfile
            self.config.update()
        return this_fit, fit_params

    def prepare_histogram(self, pars, method=0, calo=0, asym=None):
        if(self.h is None):
            self.apply_pileup_correction()

        if(method == 0):
            hi = self.h.t_method(pars['t_threshold_low'], pars['t_threshold_high'])
        elif(method == 1):
            if asym is None:
                raise ValueError("Please provide asymmetry callable")
            hi = self.h.a_method(asym, pars['a_threshold_low'],pars['a_threshold_high'])
        else:
            raise NotImplementedError()

        return hi  

    def prepare_fit(self,fit='5',method=0, calo=0, hi=None, **kwargs):
        fit_function, fit_pars = self.prepare_fit_function(fit)
        other_pars = self.config['fitting']
        if(hi is None):
            hi = self.prepare_histogram(other_pars, method, calo)
        
        lims = {fit_pars['names'][i]:(fit_pars['limlow'][i],fit_pars['limhigh'][i]) 
                    for i in range(len(fit_pars['names']))}
        print(f'{lims=}')
        # print(f"{fit_pars['fixed']=}")
        # print(f'{fit_pars=}')
        fixed_pars = [i for i,x in enumerate(fit_pars['fixed']) if x]
        this_fit = fitting.PyFit.from_hist(
            hi,
            fit_function,
            fit_pars['guess'],
            limits = (other_pars['fit_start'], other_pars['fit_end']),
            names = fit_pars['names'],
            par_limits = lims,
            fixed_pars = fixed_pars,
            **kwargs
        )
        
        return this_fit

    def update_fit(self,fitid:str, thisfit:fitting.PyFit, method='t', thisdict = None, write=True):
        '''update the fit information in the config file to reflect the completed fit'''
        directory = self.config['directory']
        date = get_date()
        if(write):
            output_file = os.path.join(directory, 'fits', fitid, f'completed_fit_{fitid}_{method}_{date}.pickle')
            os.system(f'mkdir -p {os.path.dirname(output_file)}')

            thisfit.write(output_file)

        fitted_params = []
        for i,x in enumerate(thisfit.m.parameters):
            fitted_params.append([
                x, thisfit.m.values[i],*thisfit.m.limits[i], thisfit.m.fixed[i]
            ])

        if(thisdict is None):
            if(type(method) is int):
                method = methods[method]
            if(method not in self.config['fitting']['fits'][fitid]):
                self.config['fitting']['fits'][fitid][method] = tomlkit.table()

            self.config['fitting']['fits'][fitid][method]['fitted_params'] = fitted_params
            self.config['fitting']['fits'][fitid][method]['fitted_errors'] = list(thisfit.m.errors)
            if(thisfit.m.fmin is not None):    
                self.config['fitting']['fits'][fitid][method]['fitted_errors'] = list(thisfit.m.errors)
            self.config['fitting']['fits'][fitid][method]['fitted_cov'] = [list(x) for x in list(np.array(thisfit.m.covariance))]
            if(write):
                self.config['fitting']['fits'][fitid][method]['file'] = output_file
            self.config['fitting']['fits'][fitid][method]['complete'] = True
            self.config['fitting']['fits'][fitid][method]['process_time'] = date 
        else:
            thisdict['fitted_params'] = fitted_params
            thisdict['fitted_errors'] = list(thisfit.m.errors)
            thisdict['fitted_cov'] = [list(x) for x in list(np.array(thisfit.m.covariance))]
            if(write):
                thisdict['file'] = output_file
            thisdict['complete'] = True
            thisdict['process_time'] = date 
            # self.config.update()
            return thisdict


        self.config.update()

    @staticmethod
    def save_object(obj, outfile='./out.pickle'):
        with gzip.open(outfile, 'wb') as fout:
            pickle.dump(obj, fout)

    @staticmethod
    def load_object(outfile='./out.pickle'):
        with gzip.open(outfile, 'rb') as fin:
            return pickle.load(fin)

    '''
        Systematic scan code
    '''

    '''
        Energy binned analysis
    '''
    def energy_binned_scan(self, scan_name='Ebinned'):
        '''function to perform an energy binned analysis, governed by the config file'''
        ebin_params = self.config['scans'][scan_name]
        fit_params = self.config['fitting']
        scan_dir = os.path.join(self.config['directory'], 'scans', scan_name)
        os.system(f'mkdir -p {scan_dir}')

        ebins = self.h.clusters.axes[1].edges
        # ebins = ebins[ebins >= ebin_params['min_E']]
        # ebins = ebins[ebins <  ebin_params['max_E']]

        ebin_bools = np.where((ebins >= ebin_params['min_E'])&(ebins < ebin_params['max_E']), True, False )
        ebin_ints = [i for i,x in enumerate(ebin_bools) if x]
        # print(ebins, len(ebins))
        # print(ebin_bools, ebin_ints)
        # return
        bin_groups = np.array_split(ebin_ints, ebin_params['n_bins'])
        # print(bin_groups)

        if(ebin_params['per_calo']):
            calos = list(range(25))
        else:
            calos = [0]

        for calo in calos:
            self.config['scans'][scan_name][str(calo)] = tomlkit.table()
            for i, bins in enumerate(bin_groups):
                # print(bins)
                self.config['scans'][scan_name][str(calo)][str(i)] = tomlkit.table()
                self.config['scans'][scan_name][str(calo)][str(i)]['ebins'] = list([int(x) for x in bins])
                self.config['scans'][scan_name][str(calo)][str(i)]['ebin_vals'] = list(ebins[bins[0]:bins[-1]+1])

                if(calo == 0):
                    this_histogram = self.h.clusters[:,bins[0]:bins[-1]+1:sum,::sum]
                else:
                    this_histogram = self.h.clusters[:,bins[0]:bins[-1]+1:sum,hist.loc(calo)]

                this_fit = self.prepare_fit(ebin_params['use_fit'], hi=this_histogram)
                # TODO: Implement setting/fixing extra parameters based on ebin specific settings
                # print(this_fit)
                if('fit_end' in ebin_params):
                    this_fit.set_limits((this_fit.limits[0],ebin_params['fit_end']))
                if('fit_start' in ebin_params):
                    this_fit.set_limits((ebin_params['fit_start'],this_fit.limits[1]))
                
                this_fit.fit()
                outfile = os.path.join(scan_dir, f'fit_calo{calo:02}_bins{i:03}.pickle')
                this_fit.write(outfile)
                self.config['scans'][scan_name][str(calo)][str(i)]['file'] = outfile
                self.config['scans'][scan_name][str(calo)][str(i)] = self.update_fit(
                    'scan', this_fit, write=False, thisdict=self.config['scans'][scan_name][str(calo)][str(i)]
                )
                # return this_fit
                # break
        self.config['scans'][scan_name][str(calo)]['complete'] = True
        self.config.update()

    def plot_e_binned_scan(self, scan_name='Ebinned', calo=0, write=True):


        directory = os.path.join(self.config['directory'], 'scans', scan_name, f'calo{calo:02}')
        # assert os.path.exists(directory)
        os.system(f'mkdir -p {directory}')

        scan_params = self.config['scans'][scan_name]
        these_params = self.config['scans'][scan_name][str(calo)]
        # print(these_params.keys())
        # print(these_params)
        npar = len(these_params['0']['fitted_params'])
        nscans = scan_params['n_bins']
        nearest_square = int(np.ceil(np.sqrt(npar)))
        fig, ax = plt.subplots(nearest_square,nearest_square, figsize=(5*nearest_square, 5*nearest_square), sharex=True)
        interps = {}
        eaxis = self.h.clusters.axes[1]
        energies = [np.mean(eaxis.centers[ these_params[f'{i}']['ebins'][0]:these_params[f'{i}']['ebins'][-1]+1 ]) for i in range(nscans)]
        energy_widths = []
        for i,x in enumerate(energies):
            bini = these_params[f'{i}']['ebins'][0]
            energy_widths.append( np.abs(x- eaxis.edges[bini] ) )

        print(energies)
        print(energy_widths)
        xs = np.linspace(np.amin(energies), np.amax(energies), 1000)
        for i in range(npar):
            axi = ax.ravel()[i]
            pari = these_params['0']['fitted_params'][i][0]
            axi.set_title(pari)

            pars     = [these_params[f'{j}']['fitted_params'][i][1] for j in range(nscans)]
            par_errs = [these_params[f'{j}']['fitted_errors'][i]    for j in range(nscans)]

            axi.errorbar(x=energies,xerr=energy_widths,y=pars, yerr=par_errs, fmt='o:')
            interps[pari] = interp1d(energies, pars, bounds_error=False)
            axi.plot(xs, interps[pari](xs), label='Interpolated')
        plt.tight_layout()

        if(write):
            plt.savefig(os.path.join(directory, 'energy_binned_results.png'))
            plt.savefig(os.path.join(directory, 'energy_binned_results.pdf'))
            interp_file = os.path.join(directory, 'energy_binned_interps.pickle')
            self.save_object(interps, interp_file)
            self.config['scans'][scan_name][str(calo)]['interp_file'] = interp_file
            self.config.update()


        return fig,ax,interps



                    


