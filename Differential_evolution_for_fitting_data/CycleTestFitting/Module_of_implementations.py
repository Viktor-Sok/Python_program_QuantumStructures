import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import timeit
import os 
from inspect import getfullargspec

def start_calculations(dataFileName, colm, fileNameIRF, darkX, darkY,
                              mut, recomb, pops, maxiter, bounds, transpose_dat):
    
    """------------------------------------------------------------------------------"""
    """---------------------Define model decay function here-------------------------"""
    def model_decay_function(T,a0,a1,a2,t1,t2):
        #assumed function of the isotropic part of a fluorescence decay
        return a0 + a1*np.exp(-t/t1)+a2*np.exp(-t/t2), int(round(T/dt))

    """------------------------------------------------------------------------------"""
    """------------------------------------------------------------------------------"""

    columns = [0]
    columns.extend(colm)
    """|||||||||||||||||Loading data to work with from  files in the .dat format||||||||||||||"""
    #Read in the experimental data
    experiment_data = pd.read_table(dataFileName,\
        header = None, usecols = columns )
    npData = experiment_data.to_numpy()

    #Read in an Instrumental Respond Function(IRF)
    IRF_data = pd.read_table(fileNameIRF,\
        header = None)
    npIRF = IRF_data.to_numpy()
    """|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"""


    """==============Some of the important global variables===================="""
    dt = (npIRF[1,0] - npIRF[0,0]) #interval for equally spaced data in time domain
    IRF = npIRF[:,1]/(np.sum(npIRF[:,1])*dt) # normalizing IRF function
    t = npData[:,0]*1000
    
    """========================================================================"""


    def calculate_isotropic_part(darkX,darkY,N):
        #Calculation of the isotropic part of a signal
        for i in range(1,N):
            if i%2 == 0:
                npData[:,i] -= darkX
            else:
                npData[:,i] -= darkY
        G_factor = np.sum(npData[:,2])/np.sum(npData[:,1])
        npData[:,2] = npData[:,2] / G_factor
        npData[:,4] = npData[:,4] / G_factor

        return (npData[:,3] + 2*npData[:,4])/3


    """******Subtract dark signal from the experimental data and get isotropic part of the signal******"""

    I_iso = calculate_isotropic_part(darkX,darkY,len(npData[0,:]))
    sigma2 = np.maximum(I_iso,np.ones(len(I_iso))) #removing all of the values which are less than one count
    """************************************************************************************************"""





    def observable_decay_function(*parameters):
        #calculation of observable in the experiment decay function by means of convolution through fft
        model_function, shiftT = model_decay_function(*parameters)
        fourier_I_model = np.fft.fft(model_function)
        IRF_shifted = np.roll(IRF,shiftT)
        fourier_IRF_shifted = np.fft.fft(IRF_shifted)
        return dt * np.fft.ifft(fourier_I_model * fourier_IRF_shifted).real


    """============ Choose the objective functions to minimize=================="""
    def least_square_roots(parameters):
        #objective function to minimize
        return np.sum((I_iso - observable_decay_function(*parameters))**2)

    def max_likelyhood_estimator(parameters):
        #objective function to minimize
        return 2*np.sum(observable_decay_function(*parameters) - sigma2) - \
               2*np.sum(sigma2*np.log(observable_decay_function(*parameters)/sigma2))
    """================================================================================="""



    def finding_the_minimum():
        #using diffirential evolution to minimize objective function
        start = timeit.default_timer()
        result = differential_evolution(max_likelyhood_estimator, bounds,  mutation = mut,
                                        recombination = recomb, maxiter = maxiter,  popsize = pops )
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Program Executed in "+str(execution_time)+ " sec")
        return result

    result = finding_the_minimum()
    optimParameters = result.x
    print("Has optimization been finished succecfully? : ",result.success)
    print("Reason for canceling calculations: ", result.message)
    print(optimParameters)
    print(bounds)
    fitFunction = observable_decay_function(*optimParameters)


    """xxxxxxxxxxxxx Creating folders and saving files xxxxxxxxxxxxxxx"""
    def createSubFolders(directory):
        os.makedirs(directory, exist_ok = True)

    def new_name(name,path, newseparator='_'):
        #allows to check wether the file with the certain name exists in the folder,
        #and if it does - creates another copy by adding number in brackets
        os.chdir(path) #side effect of this function - changing the working directory
        base, extension = os.path.splitext(name)
        i = 1
        while os.path.exists(name):
            name = base + newseparator + '(' + str(i) + ')' + extension
            i += 1
        return name
    
    #strings that are used to name output files 
    base, tag = os.path.splitext(dataFileName)
    tag = '_'.join([str(j) for j in colm])
    tag1 ='_'.join([str(j) for j in colm[2:4]])
    #creating folders and reading in the paths to these folders
    path0 = os.getcwd()
    createSubFolders('./' + base + '/'+ tag +'/')
    os.chdir('./' + base + '/')
    path1 = os.getcwd()
    os.chdir(path1 + '/'+ tag + '/')
    path2 = os.getcwd()


    #Exporting fitting curve
    fitFileName = 'Fit_' + dataFileName
    os.chdir(path1)
    if (not os.path.exists(fitFileName)):
        temp = pd.DataFrame([t],index = ['Time'])
        temp.to_csv( new_name('Fit_' + dataFileName, path1) ,sep = '\t', header = None, float_format='%.3f' )
    temp = pd.DataFrame(fitFunction,  columns = ['Fit_' + tag1]).transpose()
    temp.to_csv(fitFileName, mode='a', header= False, sep ='\t', float_format='%.3f')
    temp = pd.DataFrame(I_iso,  columns = ['Exp_' + tag1]).transpose()
    temp.to_csv(fitFileName, mode='a', header= False, sep ='\t', float_format='%.3f')

    #Calculating normalized coefficients
    sumOfCoefficients = 0
    for i in range (2, int( len(optimParameters)/2 )+1 ):
        sumOfCoefficients += optimParameters[i]

    #Preparing parameters for export
    exportParameters = np.copy(optimParameters)    
    for j in range (2, int( len(optimParameters)/2 )+1):
        exportParameters = np.append( exportParameters, optimParameters[j]/sumOfCoefficients)
        
    #Exporting parameters
    paramFileName = 'Param_' + dataFileName
    argsTemp = getfullargspec(model_decay_function).args
    args = argsTemp + [ 'n_' + j for j in argsTemp[2:int( len(optimParameters)/2 )+1] ]
    temp = pd.DataFrame(exportParameters,  columns = ['Expos_' + tag1], index = args ).transpose()
    if (not os.path.exists( paramFileName )):
        temp.to_csv(paramFileName, mode='a', sep ='\t',float_format='%.3f')
    else:
        temp.to_csv(paramFileName, mode='a', header= False, sep ='\t',float_format='%.3f')
    #Export of parameters for individual exposition in excel format
    df1 = pd.DataFrame([exportParameters], index = [tag], columns = args )
    df1.to_excel(new_name("output.xlsx", path2), float_format='%.3f') 
    """xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"""

    """++++++++++++++ Plotting Section ++++++++++++++++"""
    """++++++++++++++++++++++++++++++++++++++++++++++++"""
    plt.figure(1)
    plt.plot(t, npData[:,1],'.',markersize=2)
    plt.plot(t, npData[:,2],'.',markersize=2)
    plt.title("ZX and ZY comparison")
    plt.savefig("ZX_and_ZY.png")
    
    plt.figure(2)
    plt.plot(t, npData[:,3],'.',markersize=2)
    plt.plot(t, npData[:,4],'.',markersize=2)
    plt.title("YX and YY comparison")
    plt.savefig("YX_and_YY.png")

    plt.figure(3)
    plt.plot(t, I_iso,'.', color = 'r',markersize=2)
    plt.plot(t,fitFunction, linewidth = 0.5, color = 'b')
    plt.title("Fitting curve and experimental data")
    plt.savefig("Fitting_curve.png")
    
    plt.figure(4)    
    plt.plot(t, I_iso - fitFunction, '.',markersize=5)
    plt.title ( "Residuals")
    plt.savefig("Residuals.png")

    plt.figure(5)
    n, bins, patches = plt.hist(I_iso - fitFunction, 50, density=True, facecolor='g', alpha=0.75)
    plt.title ( "Histogramm_of_Residuals")
    plt.savefig("Histogramm_of_Residuals.png")
    plt.close('all')
    """+++++++++++++ End of Plotting Section ++++++++++++"""
    """++++++++++++++++++++++++++++++++++++++++++++++++++"""
    
    os.chdir(path1)
    if (transpose_dat):
        #Transposing the file fitFileName
        csv_input = pd.read_table(fitFileName).transpose()
        csv_input.to_csv(fitFileName, header = None, sep =  '\t')
    os.chdir(path0)
    
