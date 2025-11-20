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
                dx_fof = (
                    np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
                )
                dy_fof = (
                    np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
                )
                dz_fof = (
                    np.mean(dm_particles["x"]) - halo_properties["fof_halo_center_x"]
                )
                dx_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_x"]
                dy_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_y"]
                dz_sod = np.mean(dm_particles["x"]) - halo_properties["sod_halo_com_z"]
                dr_fof = np.linalg.norm([dx_fof, dy_fof, dz_fof])
                dr_sod = np.linalg.norm([dx_sod, dy_sod, dz_sod])
                return {"dr_fof": dr_fof, "dr_sod": dr_sod}

        collection = collection.evaluate(
                offset, 
                insert=True, 
                format="numpy",
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

You will also notice that we set :code:`format = "numpy"` in the call to :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>`. With this option set, the data will be provided to our function as a dictionary of scalars (for halo_properties) and a dictionary of numpy arrays (for dm_particles). If we had chosen instead :code:`format = "astropy"` (the default), the data would have been provided as a dictionary of astropy quantities and a dictionary of quantity arrays, respectively.

If you have a nested :py:class:`StructureCollection <opencosmo.StructureCollection>`, it will be passed to your function directly. You can still select specific columns from these datasets though:


.. code-block:: python

   oc.open("haloproperties.hdf5", "haloparticles.hdf5", "galaxyproperties.hdf5", "galaxyparticles.hdf5")

   def my_cool_function(halo_properties, dm_particles, galaxies):
        # the "galaxies" argument will be a StructureCollection
        # You can use its data directly, iterate through its galaxies
        # or further filter.
        
        # do fun stuff here.


   collection = collection.evaluate(
        offset, 
        insert=True, 
        format="numpy",
        dm_particles=["x","y","z"],
        halo_properties=[
            "fof_halo_center_x",
            "fof_halo_center_y",
            "fof_halo_center_z",
            "sod_halo_com_x",
            "sod_halo_com_y",
            "sod_halo_com_z"
        ]
        galaxies = {
            "galaxy_properties": ["gal_mass_bar", "gal_mass_star"],
            "star_particles": ["x", "y", "z"]
        }
   )

Currently, there are three use cases that are supported if you want the data to be inserted as a new column:

1. You evaluate using data from multiple datasets in the collection, and insert into the halo_properties or galaxy_properties dataset.
2. You evaluate using the data from a single dataset (chunked by structure), and insert the results into that dataset
3. You evaaluate using all the data in a single dataset, and insert into that dataset (using :py:meth:`evaluate_on_dataset <opencosmo.StructureCollection.evaluate_on_dataset>`)

Evaluating on a Single Dataset in a Structure Collection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can evaluate on individual datasets in the structure collection either in a structure-by-structure fashion, or in one go. For example, suppose you wanted to run a novel clustering algorithm on the star particles in a halo collection, but you only care about clustering within the individual halos:

.. code-block:: python

        collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5")

        def cluster_id(x, y, z):
                num_clusters = 10
                np.random.randint(0, 10, len(x))
                
        collection.evaluate(cluster_id, dataset = "star_particles", insert=True)

The coordinates of the star particles will be provided to the function on a structure-by-structure basis. Each star particle in your star_particles dataset will now have a cluster id that is local to its halo. Obviously this is not a good clustering algorithm, but you can replace it with whatever you want (k-means, DBSCA, FoF). Note that the function follows the same rules as if we we're calling :py:code:`evaluate` on the dataset itself. The required columns are determined from the function arguments.

Note that for now, we only support this type of workflow if the function only depends on data from the dataset it will be insrted into. We expect this to change in the future.


You may also want to do an evaluation on an individual dataset that without worrying about chunking by structure. Suppose you have a structure collection, and as part of a longer, more-complex analysis you want compute the perturbed mass we saw in the :ref:`previous example <Vectorizing Computations>`. You can accomplish this by instead using :py:meth:`evaluate_on_dataset <opencosmo.StructureCollection.evaluate_on_dataset>`

.. code-block:: python

        import opencosmo as oc
        from scipy.stats import norm

        collection = oc.open("haloproperties.hdf5", "haloparticles.hdf5")
        dm_vals = lambda: norm.rvs(1, 0.25, len(ds))

        def perturbed_mass(sod_halo_cdelta, sod_halo_mass, dm):
                mass_perturbation = sod_halo_mass * dm / sod_halo_cdelta
                return mass_perturbation
        
        result = collection.evaluate_on_dataset(perturbed_mass, dataset = "halo_properties", insert = True, vectorize = True, dm = dm_vals)

Notice that because we are operating on a single dataset, the format of the function looks exactly as if we were evaluating on a single dataset. If you only care about the result, you could also evaluate on the dataset directly:

.. code-block:: python

        result = collection["halo_properties"].evaluate_on_dataset(perturbed_mass, vectorize = True, dm = dm_vals)

However if you try to do the above while inserting the values

.. code-block:: python

        new_halo_properties = collection["halo_properties"].evaluate_on_dataset(perturbed_mass, insert = True, vectorize = True, dm = dm_vals)
argument to 
the return value will be a new dataset with the new :code:`perturbed_mass` column, but it will be not be part of the collection. By using :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>` on the structure collection directly, the result will be a new structure collection with the new column.


Evaluating on Lightcones and Simulation Collections
---------------------------------------------------

Using :py:meth:`Lightcone.evaluate <opencosmo.Lightcone.evaluate>` is identical to using :py:meth:`Dataset.evaluate <opencosmo.Dataset.evaluate>`. Although OpenCosmo represents lighcones internally as a collection of :py:class:`Datasets <opencosmo.Dataset>`, the details of broadcasting over these datasets are handled for you.

Using :py:meth:`SimulationCollection.evaluate <opencosmo.SimulationCollection.evaluate>` should also feel very familiar. However if you plan to provide arguments on a per-dataset basis (i.e. an extra numpy array that is used in the calculation) these arguments must be provided as a dictionary with the same keys as the names of the dataset in the :py:class:`SimulationCollection <opencosmo.SimulationCollection>`. For example:

.. code-block:: python

        collection = oc.open("haloproperties_multi.hdf5")

            def fof_px(fof_halo_mass, fof_halo_com_vx, random_value, other_value):
                return fof_halo_mass * fof_halo_com_vx * random_value / other_value

            random_data = {
                key: np.random.randint(0, 10, len(ds)) for key, ds in collection.items()
            }
            random_val = {
                key: np.random.randint(1, 100, 1) for key in collection.keys()
            }

            output = collection.evaluate(
                fof_px,
                vectorize=True,
                insert=False,
                format="numpy",
                random_value=random_data,
                other_value=random_val,
            )

A :py:class:`SimulationCollection <opencosmo.SimulationCollection>` can contain :py:class:`Datasets <opencosmo.Dataset>` or other collections. Besides arguments that are provided on a per-argument basis as discussed above, everything passed into this function will be passed directly into the :code:`evaluate` method of the underlying object.

Evaluating Without Return Values
--------------------------------

It is also possible to pass a function that returns None. Such a function could be used, for example, to produce a series of plots that are saved to disk:

.. code-block:: python

        import matplotlib.pyplot as plt
        from pathlib import Path

        halos = oc.open("haloproperties.hdf5", "sodproperties.hdf5")
        halos = halos.filter(oc.col("fof_halo_mass") > 1e14).take(10)
        output_path = Path("my_plots/profiles/")

        def plot_profiles(halo_properties, halo_profiles, output_path)
                plot_output_path = output_path / f"{halo_properties["fof_halo_tag"]}.png"
                dm_count = halo_profiles["sod_halo_bin_count"]*halo_profiles["sod_halo_bin_cdm_fraction"]
                plt.figure()
                plt.scatter(halo_profiles["sod_halo_bin_radius"], dm_count)
                plt.savefig(plot_output_path)

        halos.evaluate(
                plot_profiles,
                halo_properties=["fof_halo_tag"],
                halo_profiles=["sod_halo_bin_count, sod_halo_bin_cdm_fraction", "sod_halo_bin_radius"],
                output_path=output_path
        )

Stateful Computations
---------------------

Some computations may be *stateful*, meaning that the result of a computation on a given row in your dataset may affect the way the computation is performed on a later row. An example could include a machine learning model that is learning a distribution of halo profiles from a large number of examples.

In cases like these, the stateful part of the computation should be passed into :py:meth:`evaluate <opencosmo.StructureCollection.evaluate>` as a keyword argument, and mutated inside the provided function. OpenCosmo simply passes the keyword argument along to the eavluated function, so any mutated state will persist between function calls.


Performance Note
----------------

Evaluations on individual datasets (and lightcones) are lazy, whether processed through :code:`evaluate` or a StructureCollection's :code:`evaluate_on_dataset` method. Evaluations on StructureCollections that involve multiple datasets are performed eagerly. 
