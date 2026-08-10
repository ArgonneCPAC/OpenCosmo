import opencosmo as oc
from opencosmo.analysis import reduce
from opencosmo.analysis.plots import hist1d, hist2d, scatter, HMF

ds = oc.open('python/opencosmo/analysis/haloproperties.hdf5').filter(oc.col('sod_halo_mass') > 0).take(200)

# hist1d(ds,"sod_halo_mass")

# hist2d(ds, "sod_halo_mass", "fof_halo_mass")

# #set s=0.1 to test kwargs
# scatter(ds, "sod_halo_mass", "fof_halo_mass", s = 0.1)

# evaluate_kwargs = {"format": "numpy", "vectorize" : True, "insert" : False, "boxsize": 10, "bins": 15}
# reduce(ds, HMF, evaluate_kwargs = evaluate_kwargs)

HMF(ds, bins = 15, boxsize = 10)