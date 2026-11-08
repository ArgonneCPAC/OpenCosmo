``StructureCollection.evaluate`` can now take a function whose arguments are column names. The
dataset each column belongs to is inferred, and a column found in more than one dataset raises a
``ValueError``. The explicit form, with dataset names as arguments, is unchanged.
