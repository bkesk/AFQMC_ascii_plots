import numpy as np


def make_poly(x, coeff, n=None):
    '''
    return polynomial based on 'coeff' (degree is infered from shape of coeff)
    '''
    N = coeff.shape
    
    
    if n is None:
        dom = x
    else:
        dom = np.linspace(x[0],x[-1],n)
    
    poly = np.zeros(dom.shape)
    
    # note: step through coeff backwards! coeff is in descending order!
    for i,C in enumerate(coeff[::-1]):
        poly += C*np.power(dom,i)
    return dom, poly

def poly_uncertainty(cov, df):
    '''
    compute sigma_f(x) for a regression given the covariance matrix and partial derivatives of the fit function
    (on a chosen x interval)
    
    Inputs:
     - cov: covariance matrix
     - df: 2-d array containing partial derivates with respect to fit parameters as a function of x. 
           First dimension corresponds to parameter index, second corresponds to the chosen x domain
    
    '''
    #(sigma_f)^2 (x) = [\Sum_ (df/dc1)^2 (sigma_1)^2 + (df/dc2)^2 (sigma_2)^2 + cross term sigma_1,sigma_2 + ... ]
    #   return sigma_f (x) 
    
    
    n_param = cov.shape[0] # numer of params
    sigma2 = np.zeros(df.shape[1]) # x domain
    
    for i, df_i in enumerate(df):
        for j, df_j in enumerate(df):
            sigma2 += np.dot(df_i,df_j)*cov[i,j]
    
    return np.sqrt(sigma2)
    
def get_df_poly(x,degree):
    '''
    compute and return partial derivatives of polynomial with respect to expansion coeffs
    
    Inputs:
    - x : 1-D array containing x domain for derivatives
    - dgree: degree of polynomial
    
    Returns:
    - df : 2-D array containing partial derviatives (wrt expansion coeffs), as a function of x. 
          1st dimension corresponds to the coeff index (increasing order) and 2nd to the x index
    '''
    df = np.zeros((degree+1,x.shape[0]))
    for i in range(degree+1):
        df[i] = np.power(x,i)
    
    return df
    
def polyfit(x, y, dy, degree, n=None, extrap_point=None):
    '''
    returns a dictionary containing polynomial fit data.
    
    Inputs:
        x: the data x-values
        y: the data to be fit
        dy: the Gaussian uncercainty of each data point
        degree: polynimail degree
        n: number of points for returned polynomail
    
    Returns:
        results: a python dictionary with the following keys:
            coeffs: the fit coefficients
            cov: the covariance matrix
            func: the fit function plotted for all x-values or
                  over the range of x with 'n' points if given
    '''
    
    results = {}
    
    if dy is not None:
        w = 1 / (dy**2) # weights for the regression
        coeffs, cov = np.polyfit(x=x, y=y, deg=degree, w=w, full=False, cov='unscaled')
    else:
        coeffs, cov = np.polyfit(x=x, y=y, deg=degree, full=False, cov=True)
        
    # Polynomial Coefficients
    results['coeffs'] = coeffs
    results['cov'] = cov
    if n is None:
        results['func'] = make_poly(x, coeffs)
    else:
        results['dom'], results['func'] = make_poly(x, coeffs, n)
    
    print(cov)
    
    # r-squared
    p = np.poly1d(coeffs)
    print(f"Equation of best fit to order {degree}: ")
    print(p)
    
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    #ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    ssres = np.sum((y - yhat)**2)
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    #results['Rsqr'] = ssreg / sstot
    results['Rsqr'] = 1 - (ssres / sstot)
    
    results['y_pred'] = yhat
    
    if extrap_point is not None:
        results['extrap'] = p(extrap_point)
    
    return results


def AFQMC_FScorrection(ax, qmcfile, ldafile, dense_ldafile, fc_file, madelung_file, label=None, fmt=None, fit_fmt=None, x0=0.0):

    # get data
    x,afqmc,err = np.loadtxt(qmcfile, unpack=True)
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    fc_energy = np.loadtxt(fc_file, unpack=True)
    madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + lda_dense - lda_single
    
    afqmc_cor = afqmc + corr
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    ax.errorbar(x, afqmc_cor, yerr=err, fmt=fmt, label=label)

    print("X0 set to :", x0)
    result = polyfit(x-x0, afqmc_cor,err,degree=4,n=101)

    X = result['dom'] + x0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    ax.plot(X,Y,fit_fmt)
    
def AFQMC_FScorrection2(ax, qmcfile, ldafile, dense_ldafile, fc_file, madelung_file, label=None, fmt=None, fit_fmt=None, x0=0.0):

    # get data
    x,afqmc,err = np.loadtxt(qmcfile, unpack=True)
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + lda_dense - lda_single
    
    afqmc_cor = afqmc + corr
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    ax.errorbar(x, afqmc_cor, yerr=err, fmt=fmt, label=label)

    print("X0 set to :", x0)
    result = polyfit(x-x0, afqmc_cor,err,degree=4,n=101)

    X = result['dom'] + x0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    ax.plot(X,Y,fit_fmt)
    

def AFQMC_FScorrection3(ax, qmcfile, ldafile, dense_ldafile, fc_file, madelung_file, label=None, fmt=None, fit_fmt=None, x0=0.0, n_single=2, n_dense=2):

    # get data
    x,afqmc,err = np.loadtxt(qmcfile, unpack=True)
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = afqmc + corr
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    ax.errorbar(x, afqmc_cor, yerr=err, fmt=fmt, label=label)

    print("X0 set to :", x0)
    result = polyfit(x-x0, afqmc_cor,err,degree=2,n=101)

    X = result['dom'] + x0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    print("Cov. matrix = ", cov)
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    ax.plot(X,Y,fit_fmt)
    
    return x,afqmc_cor, err, X,Y

def poly_2(x,x0,a,b,c):
    result = a*np.power(x-x0,2) + b*np.power(x-x0,1) + c
    return result.flatten()

def AFQMC_FScorrection3_2(ax, qmcfile, ldafile, dense_ldafile, fc_file, madelung_file, label=None, fmt=None, fit_fmt=None, x0=0.0, n_single=2, n_dense=2, bounds=None, guess=None):

    from scipy.optimize import curve_fit
    
    # get data
    x,afqmc,err = np.loadtxt(qmcfile, unpack=True)
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = afqmc + corr
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    ax.errorbar(x, afqmc_cor, yerr=err, fmt=fmt, label=label)

    #print("X0 set to :", x0)
    #result = polyfit(x-x0, afqmc_cor,err,degree=2,n=101)

    if guess is None:
        guess = (x0,0.1,1.0e-4,np.amin(afqmc_cor))
    
    popt, cov = curve_fit(poly_2, x, afqmc_cor, sigma=err, p0=guess)
    
    coeffs = popt[1:]
    x0_fit = popt[0]
    
    print("coeffs =", coeffs)
    print("x0_fit", x0_fit)
    
    X,Y = make_poly(x - x0_fit, coeffs, n=101)
    
    sigma = np.sqrt(np.diagonal(cov))
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    # r-squared
    p = np.poly1d(coeffs)
    print(f"Equation of best fit: ")
    print(p)
    
    # fit values, and mean
    y = afqmc_cor
    yhat = p(x - x0_fit)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    Rsqr = ssreg / sstot
    
    print("R^2 = ", Rsqr)
    
    ax.plot(X+x0_fit,Y,fit_fmt)
    
    return x,afqmc_cor, err, X,Y

def AFQMC_FScorrection4(ax, qmcfile, ldafile, dense_ldafile, fc_file, madelung_file, label=None, fmt=None, fit_fmt=None, x0=0.0, n_single=2, n_dense=2):

    # get data
    x,afqmc,err = np.loadtxt(qmcfile, unpack=True)
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    x,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    # LDA inputs are in Ry from Quantum Espresso~ convert to Ha!
    lda_single = 0.5*lda_single
    lda_dense = 0.5*lda_dense
    
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections 
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = afqmc + corr
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    ax.errorbar(x, afqmc_cor, yerr=err, fmt=fmt, label=label)

    print("X0 set to :", x0)
    result = polyfit(x-x0, afqmc_cor,err,degree=2,n=101)

    X = result['dom'] + x0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    ax.plot(X,Y,fit_fmt)
    
    return x,afqmc_cor, err, X,Y


def AFQMC_FScorrection4_2(ax, qmcfile, ldafile, dense_ldafile, fc_file, madelung_file, label=None, fmt=None, fit_fmt=None, x0=0.0, n_single=2, n_dense=2, bounds=None, guess=None):

    from scipy.optimize import curve_fit
    
    # get data
    x,afqmc,err = np.loadtxt(qmcfile, unpack=True)
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    x,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    # LDA inputs are in Ry from Quantum Espresso~ convert to Ha!
    lda_single = 0.5*lda_single
    lda_dense = 0.5*lda_dense
    
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = afqmc + corr
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    ax.errorbar(x, afqmc_cor, yerr=err, fmt=fmt, label=label)

    #print("X0 set to :", x0)
    #result = polyfit(x-x0, afqmc_cor,err,degree=2,n=101)

    if guess is None:
        guess = (x0,0.1,1.0e-4,np.amin(afqmc_cor))
    
    popt, cov = curve_fit(poly_2, x, afqmc_cor, sigma=err, p0=guess)
    
    coeffs = popt[1:]
    x0_fit = popt[0]
    
    X,Y = make_poly(x - x0_fit, coeffs, n=101)
    
    sigma = np.sqrt(np.diagonal(cov))
    
    print("coeffs =", coeffs)
    print("x0_fit", x0_fit, f"({sigma[0]})")
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    # r-squared
    p = np.poly1d(coeffs)
    print(f"Equation of best fit: ")
    print(p)
    
    # fit values, and mean
    y = afqmc_cor
    yhat = p(x - x0_fit)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    Rsqr = ssreg / sstot
    
    print("R^2 = ", Rsqr)
    
    ax.plot(X+x0_fit,Y,fit_fmt)
    
    return x,afqmc_cor, err, X,Y

def err(e1,e2):
    return np.sqrt(e1**2 + e2**2)

def joint_err(e1,e2):
    return np.sqrt(e1**2 + e2**2)


def deltaSOC(ax, sf_file, soc_file, label=None, fmt='', error=True, scale=1.0, markerstyle={}):

    # get data
    if error:
        x,afqmc_sf,err_sf = np.loadtxt(sf_file, unpack=True)
        x,afqmc_soc,err_soc = np.loadtxt(soc_file, unpack=True)
    else:
        x,afqmc_sf = np.loadtxt(sf_file, unpack=True)
        x,afqmc_soc = np.loadtxt(soc_file, unpack=True)
        err_sf=err_soc=np.zeros(x.shape)
    
    delta = afqmc_soc - afqmc_sf
    err_delta = err(err_sf,err_soc)
    
    if label is None:
        label = '$\Delta_{SOC}$'
    
    # plot
    ax.errorbar(x, delta*scale, yerr=err_delta*scale, fmt=fmt, label=label,**markerstyle)

    return delta, err_delta

def eq_bulk_modulus_2nd(C2,V):
    '''
    bulk modulus for a 4th polynomial fit with V origin at V0!
    '''
    bCube = 6.748376012323316
    pa = 2.9421015697E+13
    Gpa = 1.0E-9 * pa

    #1 eV/Angstrom3 = 160.21766208 GPa
    fac = 160.21766208 # is this correct?
    
    B = 2*C2*V #+ 6*C3*V*(V-V0) + 12*C4*V*np.power(V-V0,2) # assuming input in eV / Ang.^3
    
    B = B*fac
           
    return B
    
#def eq_bulk_modulus_4th(C2,C3,C4,V,V0):
def eq_bulk_modulus_4th(C4,C3,C2,C1,C0,V,V0):
    '''
    bulk modulus for a 4th polynomial fit with V origin at V0!
    '''
    bCube = 6.748376012323316
    pa = 2.9421015697E+13
    Gpa = 1.0E-9 * pa

    #1 eV/Angstrom3 = 160.21766208 GPa
    fac = 160.21766208 # is this correct?
    
    B = 2*C2*V + 6*C3*V*(V-V0) + 12*C4*V*np.power(V-V0,2) # assuming input in eV / Ang.^3
    
    B = B*fac
    return B

def AFQMC_FScorrection_Vol(ax, volumefile, mbfile, ldafile, dense_ldafile, fc_file, madelung_file, extra_corr=None, label=None, fmt=None, fit_fmt=None, V0=0.0, n_single=2, n_dense=2, is_ry=False,
                          align_min=False, stoch_mb=True,fac=1.0):

    
    
    bCube = 6.748376012323316 # Bohr^3 / Ang^3
    eV = 27.211399
    
    # get data
    
    _,volume = np.loadtxt(volumefile, unpack=True)
    
    volume = (volume / bCube)*( 1 / n_single) # convert to Ang^3 / atom
    
    if stoch_mb:
        x,afqmc,err = np.loadtxt(mbfile, unpack=True)
    else:
        x,afqmc = np.loadtxt(mbfile, unpack=True)
    
    afqmc=fac*afqmc
    
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    try:
        x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    except:
        x,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    if is_ry:
        lda_single = lda_single*0.5
        lda_dense = lda_dense*0.5

    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = eV*(afqmc + corr) / n_single # now, we are in energy per atom
    
    if label is None:
        label = 'AFQMC'
    
    # plot
    print(x)
    print(volume)
    
    # add in extra correction if given
    if extra_corr is not None:
        afqmc_cor+=extra_corr

    print("V0 set to :", V0)
    if stoch_mb:
        result = polyfit(volume-V0, afqmc_cor, dy=eV*err/n_single, degree=2, n=2001)
    else:
        result = polyfit(volume-V0, afqmc_cor, dy=None, degree=2, n=2001)
        
    V = result['dom'] + V0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    
    # get the roots of the derivative:
    index_min = np.argmin(Y)
    minV = V[index_min]
    print("\n [+] Minimumum of fit potential: ", minV)
    print("\n [+] Eq. Bulk modulus",eq_bulk_modulus_2nd(coeffs[0],minV), " GPa")
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} V^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    if stoch_mb:
        plot_errs = err*eV/n_single
    else:
        plot_errs = None

    if align_min:
        ax.plot(V,Y - Y[index_min],fit_fmt)
        ax.errorbar(volume, afqmc_cor - Y[index_min], yerr=plot_errs, fmt=fmt, label=label)
    else:
        ax.plot(V,Y,fit_fmt)
        ax.errorbar(volume, afqmc_cor, yerr=plot_errs, fmt=fmt, label=label)
        
    return volume,afqmc_cor, plot_errs, V,Y


def model_EOS(x, B, V0, C, n=101):
    #1 eV/Angstrom3 = 160.21766208 GPa
    fac = 160.21766208 # is this correct?
    
    B = B/fac
    
    X = np.linspace(x[0],x[-1],n)
    K = B / (2*V0)
    
    Y = K*np.power(X-V0,2) + C

    return X,Y


######## COHESIVE ENERGY ###########
def E_cohesive(ax, volumefile, mbfile, E_atom, ldafile, dense_ldafile, fc_file, madelung_file, extra_corr=None,extra_err=None,label=None, fmt=None, fit_fmt=None, V0=0.0, n_single=2, n_dense=2, is_ry=False,
                          align_min=False, stoch_mb=True,scale_mb=1.0,fill_color='black',alpha=0.2,nSigma=1):
    
    bCube = 6.748376012323316 # Bohr^3 / Ang^3
    eV = 27.211399
    
    # get data
    
    _,volume = np.loadtxt(volumefile, unpack=True)
    
    volume = (volume / bCube)*( 1 / n_single) # convert to Ang^3 / atom
    
    if stoch_mb:
        x,afqmc,err = np.loadtxt(mbfile, unpack=True)
    else:
        x,afqmc = np.loadtxt(mbfile, unpack=True)
    
    afqmc=scale_mb*afqmc
    
    # FS correction
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    try:
        x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    except:
        x,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    if is_ry:
        lda_single = lda_single*0.5
        lda_dense = lda_dense*0.5

    # Constant Hamiltonian terms
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = eV*(afqmc + corr) / n_single # now, we are in units of eV / atom
    
    # add in extra correction if given ( units of eV / atom !! )
    if extra_corr is not None:
        afqmc_cor+=extra_corr
    
    if extra_err is not None:
        err = joint_err(err,extra_err)
    
    if label is None:
        label = 'AFQMC'
    
    ## shift to relative energy scale (E_atom as reference)
    afqmc_cor = afqmc_cor - E_atom*eV
    
    print("V0 set to :", V0)
    if stoch_mb:
        result = polyfit(volume-V0, afqmc_cor, dy=eV*err/n_single, degree=2, n=2001)
    else:
        result = polyfit(volume-V0, afqmc_cor, dy=None, degree=2, n=2001)
        
    V = result['dom'] + V0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
       
    # compute simga_df
    df = get_df_poly(V-V0,2)
    sigma_f = poly_uncertainty(cov, df)    
    
    # get the roots of the derivative:
    index_min = np.argmin(Y)
    minV = V[index_min]
    print("\n  [+] Given atomic energy (refernce energy) is ", E_atom*eV, " eV /atom")
    print("  [+] Minimumum volume of fit potential", minV, " Ang.^3")
    print("  [+] Cohesive Energy ", -Y[index_min], f"({sigma_f[index_min]}) eV / atom")
    print("  [+] Eq. Bulk modulus",eq_bulk_modulus_2nd(coeffs[0],minV), " GPa\n")
    
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} V^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    if stoch_mb:
        plot_errs = err*eV/n_single
    else:
        plot_errs = None

    
    
    # Debugging the set of primitive monomials
    #ax.plot(V, df[0],label='df/dC0')
    #ax.plot(V, df[1],label='df/dC1')
    #ax.plot(V, df[2],label='df/dC2')
    
    if align_min:
        ax.plot(V,Y - Y[index_min],fit_fmt)
        ax.errorbar(volume, afqmc_cor - Y[index_min], yerr=plot_errs, fmt=fmt, label=label)
        if stoch_mb:
            ax.fill_between(V,Y-Y[index_min]-sigma_f*nSigma,Y-Y[index_min]+sigma_f*nSigma,alpha=alpha,color=fill_color,edgecolor='black',linewidth=2)

    else:
        ax.plot(V,Y,fit_fmt)
        ax.errorbar(volume, afqmc_cor, yerr=plot_errs, fmt=fmt, label=label,linewidth=2)
        if stoch_mb:
            ax.fill_between(V,Y - sigma_f*nSigma,Y+sigma_f*nSigma,alpha=0.4,color=fill_color,edgecolor='black')
        
        
    return volume,afqmc_cor, plot_errs, V,Y


######## COHESIVE ENERGY (vs lattice constant!) ###########

def A7vol(a0, alpha, debug=False):
    # Volume of A7 primitive cell - see Needs PHYSICAL REVIEW B VOLUME 33, NUMBER 6, 3778 (1986)
    
    # compute rho
    alpha_rad = alpha * (np.pi / 180)
     
    rho = np.arcsin((2/np.sqrt(3))*np.sin(alpha_rad/2))
   
    # compute c/a
    c_to_a = np.sqrt(3)*(np.cos(rho)/np.sin(rho))
    
    # compte a (not lattice parameter!! see Needs 1986 )
    a = (a0 * np.sqrt(3)) / np.sqrt(1 + (c_to_a**2 / 3))
        
    if debug:
        print(f" [A7] alpha = {alpha} (degrees) = {alpha_rad} radians " )
        print(" [A7] rho = ", rho)
        print(" [A7] c/a = ", c_to_a)
        print(" [A7] a = ", a)
        
    V = c_to_a * (a**3 / (4*np.sqrt(3)))
    return V
                    

def A7latConst(V, alpha, debug=False):
    # Volume of A7 primitive cell - see Needs PHYSICAL REVIEW B VOLUME 33, NUMBER 6, 3778 (1986)
    
    # compute rho
    alpha_rad = alpha * (np.pi / 180)
    
    rho = np.arcsin((2/np.sqrt(3))*np.sin(alpha_rad/2))
    
    # compute c/a
    c_to_a = np.sqrt(3)*(np.cos(rho)/np.sin(rho))
    
    # compte a (not lattice parameter!! see Needs 1986 )
    a = np.cbrt((4*np.sqrt(3)/ c_to_a)*V)
    
    if debug:
        print(f" [A7] alpha = {alpha} (degrees) = {alpha_rad} radians " )
        print(" [A7] rho = ", rho)
        print(" [A7] c/a = ", c_to_a)
        print(" [A7] a = ", a)
        
    a0 = a*np.sqrt(1/3 + (c_to_a**2 / 9))
    return a0
    
def E_cohesive_vsA(ax, volumefile, mbfile, E_atom, ldafile, dense_ldafile, fc_file, madelung_file, extra_corr=None,extra_err=None,label=None, fmt=None, fit_fmt=None, V0=0.0, n_single=2, n_dense=2, is_ry=False,
                          align_min=False, stoch_mb=True,scale_mb=1.0,fill_color='black',alpha=0.2,nSigma=1,
                           angle=57.37):
    
    n_fit = 2001
    
    bCube = 6.748376012323316 # Bohr^3 / Ang^3
    eV = 27.211399
    
    # get data
    
    _,volume = np.loadtxt(volumefile, unpack=True)
    
    volume = (volume / bCube)*( 1 / n_single) # convert to Ang^3 / atom
    
    if stoch_mb:
        x,afqmc,err = np.loadtxt(mbfile, unpack=True)
    else:
        x,afqmc = np.loadtxt(mbfile, unpack=True)
    
    afqmc=scale_mb*afqmc
    
    # FS correction
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    try:
        x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    except:
        x,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    if is_ry:
        lda_single = lda_single*0.5
        lda_dense = lda_dense*0.5

    # Constant Hamiltonian terms
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = eV*(afqmc + corr) / n_single # now, we are in units of eV / atom
    
    # add in extra correction if given ( units of eV / atom !! )
    if extra_corr is not None:
        afqmc_cor+=extra_corr
        
    if extra_err is not None:
        err = joint_err(err,extra_err)
    
    if label is None:
        label = 'AFQMC'
    
    ## shift to relative energy scale (E_atom as reference)
    afqmc_cor = afqmc_cor - E_atom*eV
    
    degree = 2
    print("V0 set to :", V0)
    if stoch_mb:
        result = polyfit(volume-V0, afqmc_cor, dy=eV*err/n_single, degree=degree, n=n_fit)
    else:
        result = polyfit(volume-V0, afqmc_cor, dy=None, degree=degree, n=n_fit)
        
    V = result['dom'] + V0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
       
    # compute simga_df
    df = get_df_poly(V-V0,degree)
    sigma_f = poly_uncertainty(cov, df)    
    
    # get the roots of the derivative:
    index_min = np.argmin(Y)
    minV = V[index_min]
    print("\n  [+] Given atomic energy (refernce energy) is ", E_atom*eV, " eV /atom")
    print("  [+] Eq. volume of fit potential", minV, " Ang.^3")
    print("  [+] Eq. lattice constant of fit potential ", A7latConst(minV,angle), " Ang.^3")
    print("  [+] Cohesive Energy ", -Y[index_min], f"({sigma_f[index_min]}) eV / atom")
    print("  [+] Eq. Bulk modulus",eq_bulk_modulus_2nd(coeffs[0],minV), " GPa\n")
    
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} V^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    if stoch_mb:
        plot_errs = err*eV/n_single
    else:
        plot_errs = None

    

    
    if align_min:
        ax.plot(A7latConst(V,angle),Y - Y[index_min],fit_fmt)
        ax.errorbar(A7latConst(volume,angle), afqmc_cor - Y[index_min], yerr=plot_errs, fmt=fmt, label=label)
        if stoch_mb:
            ax.fill_between(A7latConst(V,angle),Y-Y[index_min]-sigma_f*nSigma,Y-Y[index_min]+sigma_f*nSigma,alpha=alpha,color=fill_color,edgecolor='black',linewidth=2)

    else:
        ax.plot(A7latConst(V,angle),Y,fit_fmt)
        ax.errorbar(A7latConst(volume,angle), afqmc_cor, yerr=plot_errs, fmt=fmt, label=label,linewidth=2)
        if stoch_mb:
            ax.fill_between(A7latConst(V,angle),Y - sigma_f*nSigma,Y+sigma_f*nSigma, alpha=alpha,color=fill_color,edgecolor='black')
        
        
    return volume,afqmc_cor, plot_errs, A7latConst(V,angle),Y


def E_cohesive_vsA_quart(ax, volumefile, mbfile, E_atom, ldafile, dense_ldafile, fc_file, madelung_file, extra_corr=None,extra_err=None,label=None, fmt=None, fit_fmt=None, V0=0.0, n_single=2, n_dense=2, is_ry=False,
                          align_min=False, stoch_mb=True,scale_mb=1.0,fill_color='black',alpha=0.2,nSigma=1,
                           angle=57.37):
    
    n_fit = 2001
    
    bCube = 6.748376012323316 # Bohr^3 / Ang^3
    eV = 27.211399
    
    # get data
    
    _,volume = np.loadtxt(volumefile, unpack=True)
    
    volume = (volume / bCube)*( 1 / n_single) # convert to Ang^3 / atom
    
    if stoch_mb:
        x,afqmc,err = np.loadtxt(mbfile, unpack=True)
    else:
        x,afqmc = np.loadtxt(mbfile, unpack=True)
    
    afqmc=scale_mb*afqmc
    
    # FS correction
    x,lda_single = np.loadtxt(ldafile, unpack=True)
    try:
        x,_,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    except:
        x,lda_dense = np.loadtxt(dense_ldafile, unpack=True)
    
    if is_ry:
        lda_single = lda_single*0.5
        lda_dense = lda_dense*0.5

    # Constant Hamiltonian terms
    _,fc_energy = np.loadtxt(fc_file, unpack=True)
    _,madelung =  np.loadtxt(madelung_file, unpack=True)
    
    # apply corrections
    corr = fc_energy + madelung + n_single*(lda_dense/n_dense - lda_single/n_single)
    
    afqmc_cor = eV*(afqmc + corr) / n_single # now, we are in units of eV / atom
    
    # add in extra correction if given ( units of eV / atom !! )
    if extra_corr is not None:
        afqmc_cor+=extra_corr
        
    if extra_err is not None:
        err = joint_err(err,extra_err)
    
    if label is None:
        label = 'AFQMC'
    
    ## shift to relative energy scale (E_atom as reference)
    afqmc_cor = afqmc_cor - E_atom*eV
    
    # KE: TEST!! using degree 4!
    degree = 4
    print("V0 set to :", V0)
    if stoch_mb:
        result = polyfit(volume-V0, afqmc_cor, dy=eV*err/n_single, degree=degree, n=n_fit)
    else:
        result = polyfit(volume-V0, afqmc_cor, dy=None, degree=degree, n=n_fit)
        
    V = result['dom'] + V0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
       
    # compute simga_df
    df = get_df_poly(V-V0,degree)
    sigma_f = poly_uncertainty(cov, df)    
    
    # get the roots of the derivative:
    index_min = np.argmin(Y)
    minV = V[index_min]
    print("\n  [+] Given atomic energy (refernce energy) is ", E_atom*eV, " eV /atom")
    print("  [+] Eq. volume of fit potential", minV, " Ang.^3")
    print("  [+] Eq. lattice constant of fit potential ", A7latConst(minV,angle), " Ang.^3")
    print("  [+] Cohesive Energy ", -Y[index_min], f"({sigma_f[index_min]}) eV / atom")
    print("  [+] Eq. Bulk modulus",eq_bulk_modulus_4th(*coeffs,minV,minV), " GPa\n") # experimental!
    

    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} V^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print(result['Rsqr'])

    if stoch_mb:
        plot_errs = err*eV/n_single
    else:
        plot_errs = None

    

    
    if align_min:
        ax.plot(A7latConst(V,angle),Y - Y[index_min],fit_fmt)
        ax.errorbar(A7latConst(volume,angle), afqmc_cor - Y[index_min], yerr=plot_errs, fmt=fmt, label=label)
        if stoch_mb:
            ax.fill_between(A7latConst(V,angle),Y-Y[index_min]-sigma_f*nSigma,Y-Y[index_min]+sigma_f*nSigma,alpha=alpha,color=fill_color,edgecolor='black',linewidth=2)

    else:
        ax.plot(A7latConst(V,angle),Y,fit_fmt)
        ax.errorbar(A7latConst(volume,angle), afqmc_cor, yerr=plot_errs, fmt=fmt, label=label,linewidth=2)
        if stoch_mb:
            ax.fill_between(A7latConst(V,angle),Y - sigma_f*nSigma,Y+sigma_f*nSigma, alpha=alpha,color=fill_color,edgecolor='black')
        
        
    return volume,afqmc_cor, plot_errs, A7latConst(V,angle),Y

def comp_correction(ax,x,E1,E2,err1,err2,x0=0.0,n1=2,n2=2,degree=1,label=None,
                    fmt=None,fit_fmt=None,fill_color='black',alpha=0.2,nSigma=1,marker_style={}):
    '''compute a per atom correction defined as:
    Delta = ( E1/n1 - E2/n2)
    
    assuming input energies are in E_Ha
    
    '''
    n_fit = 2001
    
    
    Delta = (E1/n1) - (E2/n2)
    err = joint_err(err1/n1,err2/n2)
    
    Delta = Delta
    err = err
    
    result = polyfit(x-x0, Delta, dy=err, degree=degree, n=n_fit)
    
    X = result['dom'] + x0
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    
    # compute simga_df
    df = get_df_poly(X-x0,1)
    sigma_f = poly_uncertainty(cov, df)    
        
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} V^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print("R^2 = ", result['Rsqr'])
    
    if label is None:
        label = "$\Delta$"
    
    ax.plot(X,Y,fit_fmt)
    ax.errorbar(x, Delta,yerr=err,fmt=fmt,label=label,linewidth=2,**marker_style)
    ax.fill_between(X,Y-sigma_f*nSigma,Y+sigma_f*nSigma,alpha=alpha,color=fill_color,edgecolor='black')
    
    Delta_from_fit = result['y_pred']
    
    df_inputs = get_df_poly(x-x0,1)
    err_from_fit = poly_uncertainty(cov,df_inputs) 
    
    return Delta_from_fit,err_from_fit

## CBS limit

def cbs_model(a,b,X,E):
    return b + a*np.power(X,-3)

def cbs_extrap(ax, X,E,a=1.0,b=1.0,model=cbs_model, guess=None, n=101):
    
    Xinv = np.reciprocal(X)
    X3inv = np.power(Xinv,3)
    
    print(Xinv)
    print(X3inv)
    
    # show inputs
    ax.plot(X3inv,E,'ro',markersize=10, label='raw many-body results')
    
    result = polyfit(X3inv, E, dy=None, degree=1, n=n, extrap_point=0.0)
    
    Y = result['func']

    coeffs = result['coeffs']
    cov = result['cov']
    sigma = np.sqrt(np.diagonal(cov))
    
    for i,C in enumerate(coeffs[::-1]):
        print(f" C_{i} X^{i} = {C} ({sigma[-(i+1)]}) ")
    
    print("Rsqr for CBS extrap. model", result['Rsqr'])

    x3inv = np.linspace(X3inv[0], X3inv[-1], n)
    ax.plot(x3inv,Y,'r--' ,label='CBS model fit')
       
    print(" CBS limit ", result['extrap'] )
    
    return result['extrap']
    
'''
def polyfit(x, y, dy, degree, n=None):
    
    returns a dictionary containing polynomial fit data.
    
    Inputs:
        x: the data x-values
        y: the data to be fit
        dy: the Gaussian uncercainty of each data point
        degree: polynimail degree
        n: number of points for returned polynomail
    
    Returns:
        results: a python dictionary with the following keys:
            coeffs: the fit coefficients
            cov: the covariance matrix
            func: the fit function plotted for all x-values or
                  over the range of x with 'n' points if given
    
    
    results = {}
    
    if dy is not None:
        w = 1 / (dy**2) # weights for the regression
        coeffs, cov = np.polyfit(x=x, y=y, deg=degree, w=w, full=False, cov='unscaled')
    else:
        coeffs, cov = np.polyfit(x=x, y=y, deg=degree, full=False, cov=True)
        
    # Polynomial Coefficients
    results['coeffs'] = coeffs
    results['cov'] = cov
    if n is None:
        results['func'] = make_poly(x, coeffs)
    else:
        results['dom'], results['func'] = make_poly(x, coeffs, n)
    
    print(cov)
    
    # r-squared
    p = np.poly1d(coeffs)
    print(f"Equation of best fit to order {degree}: ")
    print(p)
    
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    #ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    ssres = np.sum((y - yhat)**2)
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    #results['Rsqr'] = ssreg / sstot
    results['Rsqr'] = 1 - (ssres / sstot)
    
    results['y_pred'] = yhat
    
    return results
'''




def avg_block(beta, E, err, W):
    assert W.shape == E.shape
    assert beta.shape == E.shape
    assert err.shape == E.shape
    
    Wblk = np.sum(W)
    Eblk = np.dot(E,W)/Wblk
    # the error is simply the rms error, acounting for the walker weights
    errblk = np.sqrt(np.average(err**2, weights=W))
    betablk = np.mean(beta)
    
    return betablk, Eblk, errblk, Wblk
    
def reblock(beta, E, err, W, blk_size):
    
    #print(f"{W.shape=}")
    #print(f"{E.shape=}")
    #print(f"{beta.shape=}")
    #print(f"{err.shape=}")
    
    assert W.shape == E.shape
    assert beta.shape == E.shape
    assert err.shape == E.shape
    

    
    N = E.shape[0] // blk_size

    Beta_rb = np.zeros((N))
    Erb = np.zeros((N))
    err_rb = np.zeros((N))
    Wrb = np.zeros((N))
    
    for n in range(N):
        Beta_rb[n], Erb[n], err_rb[n], Wrb[n] = avg_block(beta[n*blk_size:(n+1)*blk_size],
                                     E[n*blk_size:(n+1)*blk_size],
                                     err[n*blk_size:(n+1)*blk_size],
                                     W[n*blk_size:(n+1)*blk_size])

    return Beta_rb, Erb, err_rb, Wrb
