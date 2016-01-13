import pkgutil
import csv

def csv_to_dict(filename):
    f = pkgutil.get_data('flavio.physics', filename)
    datareader = csv.reader(f.decode('utf-8').splitlines(), dialect='excel-tab')
    res = {}
    for line in datareader:
        if len(line) == 2: # for the central values
            # do not read the results for the c parameters - they are not needed.
            if line[0].split('_')[1][0]=='c':
                continue
            res[line[0]] = float(line[1])
        elif len(line) == 3: # for the covariance
            # do not read the results for the c parameters - they are not needed.
            if line[0].split('_')[1][0]=='c' or line[1].split('_')[1][0]=='c':
                continue
            res[(line[0],line[1])] = float(line[2])
    return res

def ffpar_dict(process, file_res, file_cov):
    res = {('formfactor',process,k):v for k, v in csv_to_dict(file_res).items()}
    cov = {('formfactor',process,k):v for k, v in csv_to_dict(file_cov).items()}
    return res, cov

ffpar_lattice = {}
ffpar_lattice.update(ffpar_dict('B->K*',
                     'data/arXiv-1501-00367v2/av_sl_results.d',
                     'data/arXiv-1501-00367v2/av_sl_covariance.d')[0])
ffpar_lattice.update(ffpar_dict('B->K*',
                     'data/arXiv-1501-00367v2/t_sl_results.d',
                     'data/arXiv-1501-00367v2/t_sl_covariance.d')[0])
ffpar_lattice.update(ffpar_dict('Bs->phi',
                     'data/arXiv-1501-00367v2/av_ss_results.d',
                     'data/arXiv-1501-00367v2/av_ss_covariance.d')[0])
ffpar_lattice.update(ffpar_dict('Bs->phi',
                     'data/arXiv-1501-00367v2/t_ss_results.d',
                     'data/arXiv-1501-00367v2/t_ss_covariance.d')[0])
ffpar_lattice.update(ffpar_dict('Bs->K*',
                     'data/arXiv-1501-00367v2/av_ls_results.d',
                     'data/arXiv-1501-00367v2/av_ls_covariance.d')[0])
ffpar_lattice.update(ffpar_dict('Bs->K*',
                     'data/arXiv-1501-00367v2/t_ls_results.d',
                     'data/arXiv-1501-00367v2/t_ls_covariance.d')[0])
