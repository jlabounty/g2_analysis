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
import pandas

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
    def __init__(self, config_file, verbose=False) -> None:
        self.h = None
        self.verbose=verbose
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
            if(self.verbose):
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
        
        method_found = False
        if(type(method) is int):
            method = methods[method]
        if(method in these_params):
            if(these_params[method]['complete']): # unless the fit was already completed
                method_found = True
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
            '''dont do this if we have the complete params'''
            if(not method_found):
                additional_params = self.get_fit_params(str(these_params['inherit_from']), method)
                for x in di:
                    di[x] = additional_params[x] + di[x]

        return di

    def prepare_fit_function(self, fit='5', method=0):
        this_config = self.config['fitting']['fits'][fit]
        if(self.verbose):
            print(this_config)
        fit_params = self.get_fit_params(fit, method)
        if(self.verbose):
            print(fit_params)
        if(this_config['complete'] or ('function_file' in this_config)):
            if(self.verbose):
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
        if(self.verbose):
            print(f"{fit_pars=}")
        if(self.verbose):
            print(f"{fit_function=}")
        if(self.verbose):
            print(f'{fit_function.params=}')
        other_pars = self.config['fitting']
        if(hi is None):
            hi = self.prepare_histogram(other_pars, method, calo)
        
        lims = {fit_pars['names'][i]:(fit_pars['limlow'][i],fit_pars['limhigh'][i]) 
                    for i in range(len(fit_pars['names']))}
        if(self.verbose):
            print(f'{lims=}')
        if(self.verbose):
            print(f"{fit_pars['fixed']=}")
        if(self.verbose):
            print(f'{fit_pars=}')
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

    def a_method_from_t_method(self, fit='5', ebinned_name='Ebinned', par_name='$A_{0}$', calo=0, 
                           hi=None, use_hist_errors=True, **kwargs):
        '''prepare an a-method fit given the corresponding t-method fit and an asym weighted histogram'''
        ebinned_config = self.config['scans'][ebinned_name][f'{calo}']
        global_fit_config = self.config['fitting']
        this_fit_config = global_fit_config['fits'][fit]['t']
        assert 'interp_file' in ebinned_config
        this_interp = self.load_object(ebinned_config['interp_file'])[par_name]

        if(hi is None):
            hi = self.h.a_method(this_interp, 
                    global_fit_config['a_threshold_low'], 
                    global_fit_config['a_threshold_high'], 
                    calo=calo)

        this_fit = self.prepare_fit(
            fit,calo=calo, hi=hi, use_hist_errors=use_hist_errors, **kwargs
        )

        return this_fit


    '''
        ***************************************************************************
        Static helper methods
        ***************************************************************************
    '''

    @staticmethod
    def save_object(obj, outfile='./out.pickle'):
        with gzip.open(outfile, 'wb') as fout:
            pickle.dump(obj, fout)

    @staticmethod
    def load_object(outfile='./out.pickle'):
        with gzip.open(outfile, 'rb') as fin:
            obj = pickle.load(fin)
        try:
            obj.re_init()
        except:
            pass # some classes won't have this function, that's ok.

        return obj

    '''
        ***************************************************************************
        ***************************************************************************
        ***************************************************************************
        Systematic scan code
        ***************************************************************************
        ***************************************************************************
        ***************************************************************************
    '''

    def make_scan_toml(self, scan_name) -> configuration.AnalysisConfig:
        scan_params = self.config['scans'][scan_name]
        outdir = os.path.join(self.config.get_directory(),'scans', scan_name)
        scan_file = os.path.join(outdir, f'{scan_name}.toml')
        os.system(f'mkdir -p {outdir}; touch {scan_file}')
        config = configuration.AnalysisConfig(str(scan_file))
        config['scans'] = tomlkit.table()
        config['scans'][scan_name] = tomlkit.table()
        config['scans'][scan_name]['outdir'] = str(outdir)
        config['scans'][scan_name]['process_time'] = get_date()

        return config

    '''
        ***************************************************************************
        Energy binned analysis
        ***************************************************************************
    '''
    def energy_binned_scan(self, scan_name='Ebinned'):
        '''function to perform an energy binned analysis, governed by the config file'''
        ebin_params = self.config['scans'][scan_name]
        fit_params = self.config['fitting']
        scan_dir = os.path.join(self.config['directory'], 'scans', scan_name)

        scan_output_config = self.make_scan_toml(scan_name)
        scan_outfile = scan_output_config.infile

        # scan_outfile = os.path.join(scan_dir, f'{scan_name}.toml')
        # os.system(f'mkdir -p {scan_dir}; touch {scan_outfile}')
        # scan_output_config = configuration.AnalysisConfig(str(scan_outfile))
        # scan_output_config['scans'] = tomlkit.table()
        # scan_output_config['scans'][scan_name] = tomlkit.table()
        self.config['scans'][scan_name]['scan_file'] = scan_outfile
        self.config.update()

        ebins = self.h.clusters.axes[1].edges
        # ebins = ebins[ebins >= ebin_params['min_E']]
        # ebins = ebins[ebins <  ebin_params['max_E']]


        ebin_bools = np.where((ebins >= ebin_params['min_E'])&(ebins < ebin_params['max_E']), True, False )
        ebin_ints = [i for i,x in enumerate(ebin_bools) if x]
        if(self.verbose):
            print(ebins, len(ebins))
        if(self.verbose):
            print(ebin_bools, ebin_ints)
        # return
        bin_groups = np.array_split(ebin_ints, ebin_params['n_bins'])
        if(self.verbose):
            print(bin_groups)

        if(ebin_params['per_calo']):
            calos = list(range(25))
        else:
            calos = [0]

        for calo in calos:
            scan_output_config['scans'][scan_name][str(calo)] = tomlkit.table()
            for i, bins in enumerate(bin_groups):
                # if(self.verbose):
                #     print(bins)
                scan_output_config['scans'][scan_name][str(calo)][str(i)] = tomlkit.table()
                scan_output_config['scans'][scan_name][str(calo)][str(i)]['ebins'] = list([int(x) for x in bins])
                scan_output_config['scans'][scan_name][str(calo)][str(i)]['ebin_vals'] = list(ebins[bins[0]:bins[-1]+1])

                if(calo == 0):
                    this_histogram = self.h.clusters[:,bins[0]:bins[-1]+1:sum,::sum]
                else:
                    this_histogram = self.h.clusters[:,bins[0]:bins[-1]+1:sum,hist.loc(calo)]

                this_fit = self.prepare_fit(ebin_params['use_fit'], hi=this_histogram)
                # TODO: Implement setting/fixing extra parameters based on ebin specific settings
                if('fit_end' in ebin_params):
                    this_fit.set_limits((this_fit.limits[0],ebin_params['fit_end']))
                if('fit_start' in ebin_params):
                    this_fit.set_limits((ebin_params['fit_start'],this_fit.limits[1]))
                
                this_fit.fit()
                outfile = os.path.join(scan_dir, f'fit_calo{calo:02}_bins{i:03}.pickle')
                this_fit.write(outfile)
                scan_output_config['scans'][scan_name][str(calo)][str(i)]['file'] = outfile
                scan_output_config['scans'][scan_name][str(calo)][str(i)] = self.update_fit(
                    'scan', this_fit, write=False, thisdict=scan_output_config['scans'][scan_name][str(calo)][str(i)]
                )
                # return this_fit
                # break
            scan_output_config['scans'][scan_name][str(calo)]['complete'] = True
        scan_output_config['scans'][scan_name]['complete'] = True
        self.config['scans'][scan_name]['complete'] = True
        scan_output_config.update()
        self.config.update()

    def make_df_e_binned_scan(self, scan_name='Ebinned'):
        '''loads the result of the energy binned analysis into a dataframe for plotting'''

        scan_config = self.config['scans'][scan_name]
        nbins = scan_config['n_bins']
        scan_results = configuration.AnalysisConfig(str(scan_config['scan_file']))['scans'][scan_name]
        # return scan_results
        dfi = []
        for calo in range(25):
            if(f'{calo}' in scan_results):
                for i in range(nbins):
                    dicti = {'calo':calo, 'point':i}
                    results = scan_results[f'{calo}'][f"{i}"]
                    bin_edge_low = results['ebin_vals']
                    bin_width = np.abs(bin_edge_low[0] - bin_edge_low[1])
                    bin_center = np.mean(np.array(bin_edge_low) + bin_width)
                    dicti['e'] = bin_center

                    bin_edge_low += [bin_edge_low[-1] + bin_width]
                    # print(bin_edge_low)
                    dicti['e_width'] = np.abs(np.min(bin_edge_low) - np.max(bin_edge_low))/2.
                    for i,x in enumerate(results['fitted_params']):
                        dicti[x[0]] = x[1]
                        dicti[f"{x[0]}_err"] = results['fitted_errors'][i]
                    

                    dfi.append(dicti)

        df = pandas.DataFrame(dfi)
        return df

    def build_interpolators_e_binned(self, scan_name='Ebinned', calo=0, write=True):
        directory = os.path.join(self.config['directory'], 'scans', scan_name, f'calo{calo:02}')
        # assert os.path.exists(directory)
        os.system(f'mkdir -p {directory}')

        scan_params  = self.config['scans'][scan_name]
        scan_file = scan_params['scan_file']

        # these_params = self.config['scans'][scan_name][str(calo)]
        scan_results = configuration.AnalysisConfig(str(scan_file))
        these_params = scan_results['scans'][scan_name][f'{calo}']
        if(self.verbose):
            print(these_params.keys())
        if(self.verbose):
            print(these_params)
        npar = len(these_params['0']['fitted_params'])
        nscans = scan_params['n_bins']
        nearest_square = int(np.ceil(np.sqrt(npar)))
        # fig, ax = plt.subplots(nearest_square,nearest_square, figsize=(5*nearest_square, 5*nearest_square), sharex=True)
        interps = {}
        eaxis = self.h.clusters.axes[1]
        energies = [np.mean(eaxis.centers[ these_params[f'{i}']['ebins'][0]:these_params[f'{i}']['ebins'][-1]+1 ]) for i in range(nscans)]
        energy_widths = []
        for i,x in enumerate(energies):
            bini = these_params[f'{i}']['ebins'][0]
            energy_widths.append( np.abs(x- eaxis.edges[bini] ) )

        if(self.verbose):
            print(energies)
        if(self.verbose):
            print(energy_widths)
        xs = np.linspace(np.amin(energies), np.amax(energies), 1000)
        for i in range(npar):
            # axi = ax.ravel()[i]
            pari = these_params['0']['fitted_params'][i][0]
            # axi.set_title(pari)

            pars     = [these_params[f'{j}']['fitted_params'][i][1] for j in range(nscans)]
            par_errs = [these_params[f'{j}']['fitted_errors'][i]    for j in range(nscans)]

            # axi.errorbar(x=energies,xerr=energy_widths,y=pars, yerr=par_errs, fmt='o:')
            # use the min/max values for those outside the interpolation range
            interps[pari] = interp1d(energies, pars, bounds_error=False, fill_value=0)
            # axi.plot(xs, interps[pari](xs), label='Interpolated')
        # plt.tight_layout()

        if(write):
            # plt.savefig(os.path.join(directory,   'energy_binned_results.png'))
            # plt.savefig(os.path.join(directory,   'energy_binned_results.pdf'))
            interp_file = os.path.join(directory, 'energy_binned_interps.pickle')
            self.save_object(interps, interp_file)
            self.config['scans'][scan_name][str(calo)] = tomlkit.table()
            self.config['scans'][scan_name][str(calo)]['complete'] = True
            self.config['scans'][scan_name][str(calo)]['interp_file'] = interp_file
            scan_results['scans'][scan_name][str(calo)]['interp_file'] = interp_file
            self.config.update()
            scan_results.update()


        return interps

    def plot_energy_binned(self,scan_name='Ebinned'):
        # scan_config = 
        df = self.make_df_e_binned_scan(scan_name)
        raise NotImplementedError


    '''
        ***************************************************************************
        Energy cut scan
        ***************************************************************************
    '''
    def make_df_energy_start_stop_scan(self, scan_name='t_method_e_scan', calo=0):
        '''turns the output of the energy_start_stop_scan function into a pandas df for plotting'''
        this_scan = self.config['scans'][scan_name]
        scan_file = this_scan['scan_file']

        scan_config = configuration.AnalysisConfig(str(scan_file))#['scans'][scan_name][str(calo)]
        if(self.verbose):
            print(scan_config)
        ding = scan_config['scans'][scan_name][str(calo)]
        scan_point_keys = [x for x in ding.keys() if 'scan_' in x]
        if(self.verbose):
            print(scan_point_keys)

        dfi = []
        for key in scan_point_keys:
            this_scan_point = ding[key]
            dicti = {
                'calo':calo,
                'elow':this_scan_point['elow'],
                'ehigh':this_scan_point['ehigh'],
            }
            for i, x in enumerate(this_scan_point['fitted_params']):
                dicti[x[0]] = x[1]
                dicti[f'{x[0]}_err'] = this_scan_point['fitted_errors'][i]
            dfi.append(dicti)

        df = pandas.DataFrame(dfi)
        return df

    def energy_start_stop_scan(self, scan_name='t_method_e_scan', write=True, rebin_e=1):
        '''does a scan over the t/a method to find the optimal energy points'''
        scan_config = self.config['scans'][scan_name]
        global_fit_config = self.config['fitting']
        method = scan_config['method']
        if(type(method) is int):
            method = methods[method]

        if(scan_config['per_calo']):
            calos = list(range(25))
        else:
            calos = [0]

        # loop over each of the calos to be evaluated
        scan_outdir = os.path.join(self.config['directory'], 'scans', scan_name)
        # scan_outfile = os.path.join(scan_outdir, f'{scan_name}.toml')
        # os.system(f'mkdir -p {scan_outdir}; touch {scan_outfile}')
        # scan_output_config = configuration.AnalysisConfig(str(scan_outfile))
        # scan_output_config['scans'] = tomlkit.table()
        # scan_output_config['scans'][scan_name] = tomlkit.table()
        scan_output_config = self.make_scan_toml(scan_name)
        scan_outfile = scan_output_config.infile
        self.config['scans'][scan_name]['scan_file'] = scan_outfile
        self.config.update()

        for calo in calos:
            # get the main histogram, without collapsing the energy axis
            if(method == 't'):
                if(calo > 0):
                    h_e = self.h.clusters[:,:,hist.loc(calo)]
                else:
                    h_e = self.h.clusters[:,:,::sum]
            elif(method == 'a'):
                ebinned_config = self.config['scans']['Ebinned'][str(calo)]
                if(not ebinned_config['complete']):
                    raise ValueError("The energy binned scan for this calo is not complete")
                interp_file = ebinned_config['interp_file']
                asym_callable = self.load_object(interp_file)['$A_{0}$']
                # asym_callable = None
                h_e = self.h.a_method(asym_callable, threshold=1000, threshold_high=3200,calo=calo, collapse_energy=False)
            else:
                raise NotImplementedError

            if(rebin_e > 1):
                h_e = h_e[:,::hist.rebin(rebin_e)]

            # create the output table for the scan and directory
            scan_output_config['scans'][scan_name][str(calo)] = tomlkit.table()
            outdir = os.path.join(self.config['directory'], 'scans', scan_name, f'{calo}')
            os.system(f'mkdir -p {outdir}')

            if(scan_config['min_start_E'] == scan_config['max_start_E']):
                ebins_low = [h_e.axes[1].edges[h_e.axes[1].index(scan_config['max_start_E'])]]
            else:
                ebins_low = h_e.axes[1].edges
                ebins_low = ebins_low[ebins_low >= scan_config['min_start_E']]
                ebins_low = ebins_low[ebins_low  < scan_config['max_start_E']]
                if(len(ebins_low) < 1):
                    raise ValueError("No bins in selected range for ebins_low")

            if(scan_config['min_end_E'] == scan_config['max_end_E']):
                ebins_high = [h_e.axes[1].edges[h_e.axes[1].index(scan_config['max_end_E'])]]
            else:
                ebins_high = h_e.axes[1].edges
                ebins_high = ebins_high[ebins_high >= scan_config['min_end_E']]
                ebins_high = ebins_high[ebins_high < scan_config['max_end_E']]
                if(len(ebins_high) < 1):
                    raise ValueError("No bins in selected range for ebins_high")

            if(self.verbose):
                print(scan_config['min_start_E'], scan_config['max_start_E'], scan_config['min_end_E'], scan_config['max_end_E'])
            if(self.verbose):
                print(f"{ebins_low=}")
            if(self.verbose):
                print(f"{ebins_high=}")
            binwidth = np.abs(h_e.axes[1].edges[0] - h_e.axes[1].edges[1])
            if(self.verbose):
                print(f"{binwidth=}")
            # return

            for binlow in ebins_low:
                for binhigh in ebins_high:
                    if(self.verbose):
                        print(f'Scanning over energy range {binlow} - {binhigh} MeV')
                    thisdict = {}
                    thisdict['elow'] = binlow
                    thisdict['ehigh'] = binhigh + binwidth
                    hi = h_e[:,hist.loc(binlow):hist.loc(binhigh):sum]
                    this_fit = self.prepare_fit(scan_config['use_fit'], hi=hi)
                    if(self.verbose):
                        print(this_fit)
                    this_fit.fit()

                    self.update_fit('', this_fit, thisdict=thisdict, write=False)
                    if(write):
                        outfile = os.path.join(outdir, f'scan_point_{binlow}_{binhigh}.pickle')
                        this_fit.write(outfile)
                        thisdict['file'] = outfile

                    scan_output_config['scans'][scan_name][str(calo)][f'scan_{binlow}_{binhigh}'] = thisdict

                #     break
                # break
            scan_output_config.update()
            self.config.update()

            # df = make_df_energy_start_stop_scan(self, scan_name, calo)
            # df.to_csv(scan_outfile.replace('pickle','csv'))

    '''
        ***************************************************************************
        Start/Stop time scan
        ***************************************************************************
    '''
        

    def do_start_time_scan(self, scan_name='start_time', write=False):
        '''do a start time scan and write the results to a toml file'''
        scan_params = self.config['scans'][scan_name]
        global_fit_params = self.config['fitting']
        fit_params = self.config['fitting']['fits'][scan_params['use_fit']]

        scan_results = make_scan_toml(self, scan_name)
        outdir = scan_results['scans'][scan_name]['outdir']
        # scan_params
        self.config['scans'][scan_name]['scan_file'] = scan_results.infile 


        if scan_params['method'] in fit_params:
            # grab the completed X-method result
            this_fit = self.load_object(fit_params[scan_params['method']]['file'])
            this_fit.re_init()
            if(self.verbose):
                print(this_fit)
        else:
            raise NotImplementedError("Please do the main fit before a start time scan")

        # fix the specified params, if any
        # TODO: Implement this.

        # determine the fit start times
        nominal_start = global_fit_params['fit_start']
        nominal_end = global_fit_params['fit_end']
        if('fit_start' in scan_params):
            nominal_start = scan_params['fit_start']
        if('fit_end' in scan_params):
            nominal_end = scan_params['fit_end']
        step = scan_params['step']
        n_points = scan_params['n_pts']
        start_times = [nominal_start + i*step for i in range(n_points)]

        # run the fits
        time_axis = this_fit.hist.axes[0]
        real_end_time = time_axis.edges[time_axis.index(nominal_end)+1]
        for i, time in enumerate(start_times):
            real_start_time = time_axis.edges[time_axis.index(time)]
            dicti = {'fit_start':real_start_time, 'fit_end':real_end_time}
            if(self.verbose):
                print(f'Modified start time to align with bin edge: {time} -> {real_start_time}')
            
            this_fit.set_limits([real_start_time, nominal_end])
            if(self.verbose):
                print(f'{this_fit.limits=}')
            this_fit.fit()

            self.update_fit('', this_fit, thisdict=dicti, write=False)
            if(write):
                this_fit.write(os.path.join(scan_results['scans'][scan_name]['outdir'], f'fit_{i:05}.pickle'))

            scan_results['scans'][scan_name][f'scan_{i:05}'] = dicti
            # if(i > 10):
            #     break

        self.config.update()
        scan_results.update()

    def make_df_from_time_scan(self, scan_name, ref_point=0) -> pandas.DataFrame:
        '''takes the result of a start/end time scan and returns a pandas dataframe for plotting'''
        scan_file = self.config['scans'][scan_name]['scan_file']
        print(f'{scan_file=}')

        scan_results = configuration.AnalysisConfig(str(scan_file))
        print(scan_results)

        these_keys = [x for x in scan_results['scans'][scan_name].keys() if 'scan_' in x]
        print(these_keys)

        dfi = []
        for i, key in enumerate(these_keys):
            dicti = {'scan_point':int(key.split('_')[1])}
            this_scan = scan_results['scans'][scan_name][key]
            dicti['fit_start'] = this_scan['fit_start']
            dicti['fit_end'] = this_scan['fit_end']
            for i,x in enumerate(this_scan['fitted_params']):
                dicti[x[0]] = x[1]
                dicti[f"{x[0]}_err"] = this_scan['fitted_errors'][i]

            dfi.append(dicti)
        return pandas.DataFrame(dfi)
                        


