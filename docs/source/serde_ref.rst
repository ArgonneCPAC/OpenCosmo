Serialization
=============

The :mod:`opencosmo.serde` package describes operations on OpenCosmo objects as
validated, JSON-serializable messages. A message records one call to a public
method. It can be sent over the wire, validated on the other side, and applied to
a live object.

Transformations on :class:`opencosmo.Dataset` and :class:`opencosmo.Lightcone`
that can be described without in-memory data have generated messages,
discriminated by their ``kind``. The live target determines which message union
is used: for example, ``take`` parses as a ``DatasetTakeMessage`` for a dataset
and a ``LightconeTakeMessage`` for a lightcone. Applying a message to a target of
another type returns an ``unsupported_message`` error.

Execution environments receive a message as decoded JSON and hold the object it
applies to. :func:`opencosmo.serde.execute_message` validates the payload against
the messages that object's type supports and applies it:

.. code-block:: python

    import json

    from opencosmo.serde import MessageError, execute_message

    payload = json.loads('{"kind": "take", "n": 10, "at": "start"}')
    result = execute_message(dataset, payload)  # same as dataset.take(10, at="start")
    if isinstance(result, MessageError):
        print(result.model_dump_json())

The two steps are also available separately: :func:`opencosmo.serde.parse_message`
returns a validated message, and :func:`opencosmo.serde.apply_message` applies one.
When a payload fails validation, the object type has no messages, or a message
belongs to another type, these functions return a :class:`opencosmo.serde.MessageError`
instead of raising. Errors raised by the operation itself, such as selecting a column
that does not exist, propagate as they would from a direct call.

Column expressions and filter masks travel as expression and mask models. These
contain only a closed set of supported operations and never accept executable code.
Regions travel as box, cone, sky-box, or HEALPix region messages.

Validation checks the same constraints the method signature declares, such as
allowed values for ``mode`` and non-negative row counts. Checks that depend on the
dataset itself, such as whether a column exists, happen when the message is
applied.

Variadic keyword objects cannot contain explicit parameter names from the live
method signature. For example, ``mode`` is reserved in selection keywords, and
``descriptions``, ``allow_overwrite`` and ``mode`` are reserved in new-column
keywords. Such collisions are rejected during parsing, before the method is called.

Lightcone messages
------------------

Lightcone messages mirror the Dataset messages for ``filter``, ``select``,
``drop``, ``sort_by``, ``take``, ``take_range``, ``take_rows``, ``bound``,
``with_new_columns`` and ``with_units``, and add ``with_redshift_range`` and
``pixel_search``. Their models are exported from :mod:`opencosmo.serde` as
``Lightcone<Operation>Message`` classes, alongside the discriminated
``LightconeMessage`` union and the ``LIGHTCONE_MESSAGE_TYPES`` tuple.

Fields preserve the public method's parameter names, so payloads differ from the
Dataset payloads where the signatures differ. In particular,
``Lightcone.with_new_columns`` stores its expressions in ``columns`` rather than
``new_columns``:

.. code-block:: python

    payload = {
        "kind": "with_new_columns",
        "descriptions": {"m2": "doubled mass"},
        "columns": {
            "m2": {
                "kind": "arithmetic",
                "operator": "multiply",
                "left": {"kind": "column", "name": "fof_halo_mass"},
                "right": {"kind": "number", "value": 2},
            }
        },
    }
    result = execute_message(lightcone, payload)

``with_redshift_range`` takes finite ``z_low`` and ``z_high`` values. HEALPix query
pixels for ``pixel_search`` use nested ordering: ``pixels`` must be a nonempty
list of non-negative integers below ``12 * nside**2``, and ``nside`` must be a
positive power of two. Pixels are converted to a one-dimensional signed ``int64``
array at application time.

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
