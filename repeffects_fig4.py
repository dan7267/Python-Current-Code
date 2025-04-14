import numpy as np
import scipy as sp

def produce_basic_statistics(y, plag):
    """Produces the initial and repeated statistics for each sub"""
    #initialise arrays
    n_subs = len(y)
    avgAmp1r1 = np.zeros((n_subs, 1))
    avgAmp2r1 = np.zeros((n_subs, 1))
    avgAmp1r2 = np.zeros((n_subs, 1))
    avgAmp2r2 = np.zeros((n_subs, 1))
    wtc1 = np.zeros((n_subs, 1))
    wtc2 = np.zeros((n_subs, 1))
    btc1 = np.zeros((n_subs, 1))
    btc2 = np.zeros((n_subs, 1))
    svm_init = np.zeros((n_subs, 1))
    svm_rep = np.zeros((n_subs, 1))

    nBins = 6

    sc_trend = np.zeros((n_subs, nBins))
    abs_ad_trend = np.zeros((n_subs, nBins))

    for sub in range(n_subs):
        pattern = y[sub]
        v = pattern.shape[1] # voxels
        n = pattern.shape[0] # trials


        if n % 4 != 0:
            raise ValueError("Assumes 4 conditions with equal trials")
        
        cond1_p = [
            pattern[:n // 4, :v],               # Rows 1 to n/4
            pattern[n // 4:n // 2, :v],         # Rows n/4 + 1 to n/2
        ]
        cond2_p = [
            pattern[n // 2:3 * n // 4, :v],     # Rows n/2 + 1 to 3n/4
            pattern[3 * n // 4:, :v],           # Rows 3n/4 + 1 to n
        ]


        # Compute means
        cond1r1 = np.mean(cond1_p[0], axis=0)
        cond2r1 = np.mean(cond2_p[0], axis=0)
        cond1r2 = np.mean(cond1_p[1], axis=0)
        cond2r2 = np.mean(cond2_p[1], axis=0)
        avgAmp1r1[sub, :] = np.mean(cond1r1, axis=0)
        avgAmp2r1[sub, :] = np.mean(cond2r1, axis=0)
        avgAmp1r2[sub, :] = np.mean(cond1r2, axis=0)
        avgAmp2r2[sub, :] = np.mean(cond2r2, axis=0)
        avgR1 = (avgAmp1r1+avgAmp2r1)/2
        avgR2 = (avgAmp1r2+avgAmp2r2)/2

        # Sorting and indexing for AMA adaptation by amplitude
        sAmp1r = np.sort(np.mean(np.hstack([cond1_p[0].T, cond1_p[1].T]), axis=1))
        ind1 = np.argsort(np.mean(np.hstack([cond1_p[0].T, cond1_p[1].T]), axis=1))

        sAmp2r = np.sort(np.mean(np.hstack([cond2_p[0].T, cond2_p[1].T]), axis=1))
        ind2 = np.argsort(np.mean(np.hstack([cond2_p[0].T, cond2_p[1].T]), axis=1))

        # Reorder based on indices
        sAmp1r1 = cond1r1[ind1]
        sAmp2r1 = cond2r1[ind2]
        sAmp1r2 = cond1r2[ind1]
        sAmp2r2 = cond2r2[ind2]

        # Compute slope
        sAmp = ((sAmp1r1 - sAmp1r2) + (sAmp2r1 - sAmp2r2)) / 2


        #Computing WC and BC
        cond1_p1_corr = (np.corrcoef(cond1_p[0].T, rowvar=False)+np.corrcoef(cond2_p[0].T, rowvar=False)) / 2
        wtc1[sub, :] = np.mean(np.mean(cond1_p1_corr, axis=0))

        cond1_p2_corr = (np.corrcoef(cond1_p[1].T, rowvar=False)+np.corrcoef(cond2_p[1].T, rowvar=False)) / 2
        wtc2[sub, :] = np.mean(np.mean(cond1_p2_corr, axis=0))

        #Only difference between matlab and python corr functions is
        #MATLAB uses columns as variables whereas NumPy uses rows. Not applicable for wtc

        pp1 = np.corrcoef(cond1_p[0].T, cond2_p[0].T, rowvar=False)
        pp11 = (cond1_p[0].T).shape[1]
        pp1 = pp1[:pp11, pp11:]

        btc1[sub, :] = np.mean(np.mean(pp1, axis=0))

        pp2 = np.corrcoef(cond1_p[1].T, cond2_p[1].T, rowvar=False)
        pp22 = (cond1_p[1].T).shape[1]
        pp2 = pp2[:pp22, pp22:]
        btc2[sub, :] = np.mean(np.mean(pp2, axis=0))

        svm_init[sub, :] = wtc1[sub, :] - btc1[sub, :]
        svm_rep[sub, :] = wtc2[sub, :] - btc2[sub, :]

        # Perform t-tests
        tval1, pval1 = sp.stats.ttest_ind(np.vstack([cond1_p[0], cond1_p[1]]), np.vstack([cond2_p[0], cond2_p[1]]), axis=0)
        tval2, pval2 = sp.stats.ttest_ind(np.vstack([cond2_p[0], cond2_p[1]]), np.vstack([cond1_p[0], cond1_p[1]]), axis=0)

        # Sorting the t-values by their absolute values
        tval_sorted_ind1 = np.argsort(np.abs(tval1))
        tval_sorted_ind2 = np.argsort(np.abs(tval2))

        # Compute means for conditions
        c1_init = np.mean(cond1_p[0], axis=0)
        c1_rep = np.mean(cond1_p[1], axis=0)
        c2_init = np.mean(cond2_p[0], axis=0)
        c2_rep = np.mean(cond2_p[1], axis=0)

        # Reorder based on sorted indices
        c1_sinit = c1_init[tval_sorted_ind1]
        c1_srep = c1_rep[tval_sorted_ind1]
        c2_sinit = c2_init[tval_sorted_ind2]
        c2_srep = c2_rep[tval_sorted_ind2]

        # Compute trends
        abs_init_trend = (c1_sinit + c2_sinit) / 2
        abs_rep_trend = (c1_srep + c2_srep) / 2
        abs_adaptation_trend = abs_init_trend - abs_rep_trend

        #Binning the AMA and AMS trends
        AA = sAmp
        AS = abs_adaptation_trend

        # Compute the percentage indices (similar to MATLAB's rounding and indexing)
        percInds = (np.round((np.arange(1, len(AA) + 1) * (nBins - 1)) / len(AA)) / (nBins - 1)) * (nBins - 1)

        for i in range(nBins):
            sc_trend[sub, i] = np.mean(AA[percInds == i], axis=0)
            abs_ad_trend[sub, i] = np.mean(AS[percInds == i], axis=0)

    return np.column_stack((avgR1, avgR2)), np.column_stack((svm_init, svm_rep)), np.column_stack((wtc1, wtc2)), np.column_stack((btc1, btc2)), np.array(sc_trend), np.array(abs_ad_trend)



def produce_slopes(y, pflag):
    """Returns the size and direction of each data feature"""
    AM, CP, WC, BC, AMA, AMS = produce_basic_statistics(y, pflag)

    def compute_slope(data):
        L = data.shape[1]
        X = np.vstack((np.arange(1, L+1), np.ones(L))).T
        pX = np.linalg.pinv(X)
        return np.dot(pX, data.T)[0]
    
    slopes_dict = {name: compute_slope(data) for name, data in zip(
        ['AM', 'WC', 'BC', 'CP', 'AMS', 'AMA'], [AM, WC, BC, CP, AMS, AMA]
    )}

    # t_results = {name: sp.stats.ttest_1samp(slope, 0) for name, slope in slopes.items()}
    slopes = [np.mean(slopes_dict[key]) for key in slopes_dict]

    return slopes

y = np.array([[[-0.9969, -1.3622],
         [ 0.7768, -1.1933],
         [ 0.8446, -0.3450],
         [ 0.1953, -0.0743],
         [-0.8318,  1.1681],
         [-0.3868,  2.1993],
         [ 2.4177, -0.5848],
         [ 0.3025,  1.4126]],

        [[-0.5642, -1.4456],
         [-0.8783,  1.2357],
         [ 0.1728, -0.6139],
         [ 0.6109,  0.1852],
         [ 0.7538, -1.7449],
         [ 1.7639,  0.8480],
         [ 0.6495, -2.4907],
         [-1.4072, -0.2315]]])

print(produce_basic_statistics(y, 1))
#MATLAB produces identical results