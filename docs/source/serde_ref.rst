Serialization
=============

The :mod:`opencosmo.serde` package describes operations on OpenCosmo objects as
validated, JSON-serializable messages. A message records one call to a public
method. It can be sent over the wire, validated on the other side, and applied to
a live object.

Every :class:`opencosmo.Dataset` transformation that can be described without
in-memory data has a message, discriminated by its ``kind``. Validate an incoming
payload with the ``DatasetMessage`` union and call ``apply`` to perform the
operation:

.. code-block:: python

    from pydantic import TypeAdapter

    from opencosmo.serde import DatasetMessage

    adapter = TypeAdapter(DatasetMessage)
    message = adapter.validate_json(
        '{"kind": "take", "n": 10, "at": "start"}'
    )
    subset = message.apply(dataset)  # same as dataset.take(10, at="start")

Column expressions and filter masks travel as expression and mask models. These
contain only a closed set of supported operations and never accept executable code.
Regions travel as box, cone, sky-box, or HEALPix region messages.

Validation checks the same constraints the method signature declares, such as
allowed values for ``mode`` and non-negative row counts. Checks that depend on the
dataset itself, such as whether a column exists, happen when the message is
applied.

Generated and hand-written code
-------------------------------

Message classes in :mod:`opencosmo.serde.messages` are generated from the public
method signatures and checked into the repository. Their fields, defaults, and
``apply`` methods mirror the method each one represents, so a change to a public
signature changes the corresponding message. After changing a signature, or the
generator configuration in ``opencosmo/serde/codegen/config.py``, regenerate the
messages:

.. code-block:: bash

    uv run python -m opencosmo.serde.codegen

CI runs the same command with ``--check`` and fails when the checked-in messages are
out of date. Review the generated diff alongside the API change.

The expression, mask, and region grammars, the conversions from wire values to live
objects, and checks that span several parameters are written by hand.

.. automodule:: opencosmo.serde
   :members:
   :imported-members:
