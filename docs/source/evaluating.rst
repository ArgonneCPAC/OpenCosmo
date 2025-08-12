Evaluating Complex Expressions on Datasets and Collections
==========================================================

Sometimes, it's possible to compute new quantities with simple algebraic combinations of columns that are already in your dataset. In these cases, you can use :py:meth:`with_new_columns <opencosmo.Dataset.with_new_columns>` to efficiently create new columns out of existing ones. However most interesting science requires more sophisticated options. Maybe you have a large number of dark matter halos and you want to find the best-fit NFW profile for each of them. Or maybe you are interested in galaxy clusters, and want to determine the most massive galaxy within each cluster-scale halo.

For these kinds of cases, OpenCosmo provides the :py:meth:`evaluate <opencosmo.Dataset.evaluate>` method. :code:`evaluate` takes a function you provide it and evaluates over all the rows in your dataset. When you're working with a :py:class:`StructureCollection <opencosmo.StructureCollection>` you can use :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>` to evalute your function over all *structures* in the collection, potentially performing complex computations that involve multiple particle species.

Evaluating on Datasets
----------------------

To evaluate a function on all rows in a single dataset, simply write a function that takes in arguments with the same name as some of the columns in your dataset and returns a dictionary of values:

.. code-block:: python

        import opencosmo as oc
        dataset = oc.open("haloproperties.hdf5")

        ds = oc.open(input_path).filter(oc.col("sod_halo_cdelta") > 0)

        def nfw(sod_halo_radius, sod_halo_cdelta, sod_halo_mass):
                r_ = np.logspace(-2, 0, 50)
                A = np.log(1 + sod_halo_cdelta) - sod_halo_cdelta / (1 + sod_halo_cdelta)

                halo_density = sod_halo_mass / (4 / 3 * np.pi * sod_halo_radius**3)
                profile = halo_density / (3 * A * r_) / (1 / sod_halo_cdelta + r_) ** 2
                return {"nfw_radius": r_ * sod_halo_radius, "nfw_profile": profile}

        ds = ds.evaluate(nfw, insert = True)

Your dataset will now include a column named "nfw_radius" with the radius values, and "nfw_profile" with the associated profile. Note that opencosmo, like astropy, supports multi-dimensional columns.
        
Additional Arguments
^^^^^^^^^^^^^^^^^^^^

If your function requires more arguments than just column names, you can pass them directly as keyword arguments to :py:meth:`evaluate <opencosmo.Dataset.evaluate>`. These arguments will be passed along to the underlying without modification. For example, suppose we wanted to perturb the true mass of a halo by some random amount to simulate the uncertainty associated with inferring masses through observation:

.. code-block:: python

        import opencosmo as oc
        from scipy.stats import norm

        ds = oc.open("haloproperties.hdf5")
        dm_pdf = lambda: norm.rvs(1, 0.25, 1)
        def perturbed_mass(sod_halo_cdelta, sod_halo_mass, dm_simulator):
                # suppose uncertainty increases with mass and decreases with concentration
                dm = dm_simulator()
                mass_perturbation = sod_halo_mass * dm / sod_halo_cdelta
                return mass_perturbation
                


        result = ds.evaluate(perturbed_mass, dm_simulator = dm_pdf)

Note that we also do not set :code:`insert = True` in the call to :py:meth:`evaluate <opencosmo.Dataset.evaluate>`. Instead, the method will simply return the resutlts to us. Additionally, our function returns a single value rather than a dictionary. As a result, the name of the column will be the same as the name of the function.

Vectorizing Computations
^^^^^^^^^^^^^^^^^^^^^^^^

Although the above example will work it involves performing the computation one row at a time, which is very inefficient. We can speed this up in two ways. First, we can generate all the random values ahead of time, rather than making a call to the random number generator at each iteration:
        
.. code-block:: python

        import opencosmo as oc
        from scipy.stats import norm

        ds = oc.open("haloproperties.hdf5")
        dm_vals = lambda: norm.rvs(1, 0.25, len(ds))

        def perturbed_mass(sod_halo_cdelta, sod_halo_mass, dm):
                mass_perturbation = sod_halo_mass * dm / sod_halo_cdelta
                return mass_perturbation
                
        result = ds.evaluate(perturbed_mass, dm = dm_vals)

The toolkit will automatically detect that dm_vals is the same length as the dataset, and break it up by row accordingly.

However this is still not very efficient. This entire computation can be vectorized by simply doing the computation with the entire columns. Because Astropy columns are just numpy arrays, standard numpy syntax will work without issue. You can request vectorization by simply setting :code:`vectorize = True` in the call to :py:meth:`evaluate <opencosmo.Dataset.evaluate>`:

.. code-block:: python

        import opencosmo as oc
        from scipy.stats import norm

        ds = oc.open("haloproperties.hdf5")
        dm_vals = lambda: norm.rvs(1, 0.25, len(ds))

        def perturbed_mass(sod_halo_cdelta, sod_halo_mass, dm):
                mass_perturbation = sod_halo_mass * dm / sod_halo_cdelta
                return mass_perturbation
        
        result = ds.evaluate(perturbed_mass, dm = dm_vals, vectorize = True)

Evaluating on Structure Collections
-----------------------------------

When working with a :py:class:`StructureCollection <opencosmo.StructureCollection>`, you can use :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>` to perform complex computations involving multiple datasets.


.. code-block:: python

        collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5").take(200)
        def offset(halo_properties, dm_particles):
                particle_data = dm_particles.get_data("numpy")
                dx_fof = (
                    np.mean(particle_data["x"]) - halo_properties["fof_halo_center_x"].value
                )
                dy_fof = (
                    np.mean(particle_data["x"]) - halo_properties["fof_halo_center_x"].value
                )
                dz_fof = (
                    np.mean(particle_data["x"]) - halo_properties["fof_halo_center_x"].value
                )
                dx_sod = np.mean(particle_data["x"]) - halo_properties["sod_halo_com_x"].value
                dy_sod = np.mean(particle_data["x"]) - halo_properties["sod_halo_com_y"].value
                dz_sod = np.mean(particle_data["x"]) - halo_properties["sod_halo_com_z"].value
                dr_fof = np.linalg.norm([dx_fof, dy_fof, dz_fof])
                dr_sod = np.linalg.norm([dx_sod, dy_sod, dz_sod])
                return {"dr_fof": dr_fof, "dr_sod": dr_sod}

        collection = collection.evaluate(
                offset, 
                insert=True, 
                dm_particles=["x","y","z"]
                halo_properties=[
                        "fof_halo_center_x",
                        "fof_halo_center_y",
                        "fof_halo_center_z",
                        "sod_halo_com_x",
                        "sod_halo_com_y",
                        "sod_halo_com_z"
                ]
        )

There are two clear differences between this example and the one with a single dataset. First, you must explicitly declare which columns you need from each of the datasets in the collection. The columns are passed as keyword arguments to :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>`. Secondly, the function that does the computation takes the names of the datasets themselves as input parameters, rather than the names of columns. This ensures you can, for example, work with multiple species of particles in a single function even if they have some of the same column names.

Evaluating on a Single Dataset in a Structure Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just because you have a structure collection doesn't mean you need to use multiple datasets for every evaluation. Suppose you have a structure collection, and as part of a longer, more-complex analysis you want compute the perturbed mass we saw in the :ref:`previous example <Vectorizing Computations>`. You can accomplish this by passing a dataset argument to :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>`

.. code-block:: python

        import opencosmo as oc
        from scipy.stats import norm

        collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
        dm_vals = lambda: norm.rvs(1, 0.25, len(ds))

        def perturbed_mass(sod_halo_cdelta, sod_halo_mass, dm):
                mass_perturbation = sod_halo_mass * dm / sod_halo_cdelta
                return mass_perturbation
        
        result = collection.evaluate(perturbed_mass, dataset = "halo_properties", insert = True, vectorize = True, dm = dm_vals)

Notice that because we are operating on a single dataset, the format of the function looks exactly as if we were evaluating on a single dataset. If you only care about the result, you could also evaluate on the dataset directly:

.. code-block:: python

        result = collection["halo_properties"].evaluate(perturbed_mass, vectorize = True, dm = dm_vals)

However if you try to do the above with an insert

.. code-block:: python

        new_halo_properties = collection["halo_properties"].evaluate(perturbed_mass, insert = True, vectorize = True, dm = dm_vals)

the return value will be a new dataset with the new :code:`perturbed_mass` column, but it will be not be part of the collection. By using :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>` on the structure collection directly, the result will be a new structure collection with the new column.



