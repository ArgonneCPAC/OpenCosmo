``Dataset.bound`` now returns the correct rows when the dataset's rows are not in on-disk
order, such as after ``sort_by``, on matched datasets, or on structure collection
children.
