import opencosmo as oc
from opencosmo.analysis import reduce
import matplotlib.pyplot as plt
import numpy as np

def hist1d(data, column_name, bins = 20):
    
    quantity = data#ds.select(column_name).get_data("numpy")
    bin_edges = np.logspace(np.log10(quantity.min()), np.log10(quantity.max()), bins)

    plt.hist(quantity, bins = bin_edges,  edgecolor='black')
    plt.xlabel(column_name)
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(f'{column_name}_hist.png')
    plt.show()
    plt.close()
    
    return

def hist2d(ds, x_column, y_column, gridsize = 50, cmap = 'Blues'):
    
    x = ds.select(x_column).get_data("numpy")
    y = ds.select(y_column).get_data("numpy")
    
    hb = plt.hexbin(x, y, gridsize=gridsize, bins='log', xscale = 'log', yscale = 'log', cmap=cmap)
    cb = plt.colorbar(hb, label='Counts')
    
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.show()
    
    return

def scatter(ds, x_column, y_column, xscale = 'log', yscale = 'log', labels = None, **kwargs):
    
    x = ds.select(x_column).get_data("numpy")
    y = ds.select(y_column).get_data("numpy")
    
    for k, val in kwargs.items():
        plt.scatter(x, y, **kwargs)
    
    if labels == None:
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        
    else:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.show()
    
    return

def reduce_hmf(sod_halo_mass, bins, boxsize):
    b = np.logspace(np.log10(sod_halo_mass).min(), np.log10(sod_halo_mass).max(), bins)
    h, bin_edge = np.histogram(sod_halo_mass, bins = b)
    
    mbins = 0.5 * (bin_edge[:-1] + bin_edge[1:])
    hmf = np.abs(h/np.diff(bin_edge)/boxsize**3)
    err = np.abs(np.sqrt(h)/np.diff(bin_edge)/boxsize**3)
    # log_err = err / (np.log(10)*hmf)
    
    plt.plot(mbins, hmf, linestyle = '--', color = 'k')
    plt.errorbar(mbins, hmf, yerr = err , color = 'k', fmt = 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Halo Mass [$M_\odot$]')
    plt.ylabel(r'Number Density $\frac{dn}{dM}$')
    plt.show()
    plt.close()
    
def HMF(ds, bins, boxsize):
    evaluate_kwargs = {"format": "numpy", "vectorize" : True, "insert" : False, "boxsize": boxsize, "bins": bins}
    reduce(ds, reduce_hmf, evaluate_kwargs = evaluate_kwargs)
    
    
    
    