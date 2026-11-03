Added ``opencosmo.serde``, which describes ``Dataset`` transformations as validated,
JSON-serializable messages. Messages are generated from the ``Dataset`` method
signatures and can be applied to a dataset with ``message.apply(dataset)``.
