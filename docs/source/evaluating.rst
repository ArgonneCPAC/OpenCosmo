Evaluating Complex Expressions on Datasets and Collections
==========================================================

Sometimes, it's possible to compute new quantities with simple algebraic combinations of columns that are already in your dataset. In these cases, you can use :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` to efficiently create new columns out of existing ones. However most interesting science requires more sophisticated options. Maybe you have a large number of dark matter halos and you want to find the best-fit NFW profile for each of them. Or maybe you are interested in galaxy clusters, and want to determine the most massive galaxy within each cluster-scale halo.

For these kinds of cases, OpenCosmo provides the :py:meth:`evaluate <opencosmo.Dataset.evaluate>` method. :code:`evaluate` takes a function you provide it and evaluates over all the rows in your dataset. When you're working with a :py:class:`StructureCollection <opencosmo.StructureCollection>` you can use :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>` to evalute your function over all *structures* in the collection, potentially performing complex computations that involve multiple particle species.

Evaluating on Datasets
----------------------

To evaluate a function on all rows in a single dataset, simply write a function that takes in arguments with the same name as some of the columns in your dataset and returns a dictionary of values:

.. code-block:: python

        import opencosmo as oc
        import numpy as np
        dataset = oc.open("haloproperties.hdf5")

        ds = oc.open(input_path).filter(oc.col("sod_halo_cdelta") > 0)

        def nfw(sod_halo_radius, sod_halo_cdelta, sod_halo_mass):
                r_ = np.logspace(-2, 0, 50)
                A = np.log(1 + sod_halo_cdelta) - sod_halo_cdelta / (1 + sod_halo_cdelta)

                halo_density = sod_halo_mass / (4 / 3 * np.pi * sod_halo_radius**3)
                profile = halo_density / (3 * A * r_) / (1 / sod_halo_cdelta + r_) ** 2
                return {"nfw_radius": r_ * sod_halo_radius, "nfw_profile": profile}

        ds = ds.evaluate(nfw)

Your dataset will now include a column named "nfw_radius" with the radius values, and "nfw_profile" with the associated profile. Note that opencosmo, like astropy, supports multi-dimensional columns.
        
        

