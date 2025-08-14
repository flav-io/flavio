import unittest
import flavio
import numpy as np


class TestBsToPhiMuMuObs(unittest.TestCase):

    def test_time_integrated_obs(self) -> None:
        """ 
        tests the predictions for the time-integrated observables for 
        Bs -> phi mumu between transversity and helicity amplitudes 
        """
        q2_points = np.append(np.linspace(0.1, 8.0, 20), np.linspace(14, 18.9, 10))

        observables = ['FL', 'AFB', 'S1s', 'S1c', 'S2s', 'S2c', 
                       'S3', 'S4', 'A5', 'A6s', 'S7', 'A8', 'A9']
        
        # get predictions for the helicity amplitudes without QCDF and subleading corrections
        flavio.config['implementation']['B->Vll angular coefficients'] = 'Helicity Amplitudes (no corrections)'
        results_helicity = {
            obs: np.array([flavio.sm_prediction(f'{obs}(Bs->phimumu)', q2) for q2 in q2_points])
            for obs in observables
        }

        # get predictions for the transversity amplitudes without QCDF and subleading corrections
        flavio.config['implementation']['B->Vll angular coefficients'] = 'Transversity Amplitudes (no corrections)'
        results_transversity = {
            obs: np.array([flavio.sm_prediction(f'{obs}(Bs->phimumu)', q2) for q2 in q2_points])
            for obs in observables
        }

        for key in observables:
            self.assertTrue(np.allclose(results_helicity[key], results_transversity[key]),
                            f'Predictions for {key} do not match between helicity and transversity amplitudes!')

        return None
    
    def test_time_dependent_obs(self) -> None:
        """ 
        tests the predictions for the time-dependent observables for 
        Bs -> phi mumu between transversity and helicity amplitudes 
        """
        q2_points = np.append(np.linspace(0.1, 8.0, 20), np.linspace(14, 18.9, 10))

        indices = ['1s', '1c', '2s', '2c', '3', '4', '5', '6s', '7', '8', '9']
        obs_names = ['K', 'W', 'H', 'Z', 'M', 'Q']
        observables = [f'{obs}{idx}' for obs in obs_names for idx in indices]

        # get predictions for the helicity amplitudes without QCDF and subleading corrections
        flavio.config['implementation']['B->Vll angular coefficients'] = 'Helicity Amplitudes (no corrections)'
        results_helicity = {
            obs: np.array([flavio.sm_prediction(f'{obs}(Bs->phimumu)', q2) for q2 in q2_points])
            for obs in observables
        }

        # get predictions for the transversity amplitudes without QCDF and subleading corrections
        flavio.config['implementation']['B->Vll angular coefficients'] = 'Transversity Amplitudes (no corrections)'
        results_transversity = {
            obs: np.array([flavio.sm_prediction(f'{obs}(Bs->phimumu)', q2) for q2 in q2_points])
            for obs in observables
        }

        for key in observables:
            self.assertTrue(np.allclose(results_helicity[key], results_transversity[key]),
                            f'Predictions for {key} do not match between helicity and transversity amplitudes!')

        return None
