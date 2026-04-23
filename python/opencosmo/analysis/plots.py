import opencosmo as oc
import matplotlib.pyplot as plt
import numpy as np

def hist1d(ds, column_name, bins = 20):
    
    quantity = ds.select(column_name).get_data("numpy")
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