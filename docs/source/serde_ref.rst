Serialization
=============

The :mod:`opencosmo.serde` package describes operations on OpenCosmo objects as
validated, JSON-serializable messages. A message records one call to a public
method. It can be sent over the wire, validated on the other side, and applied to
a live object.

Every :class:`opencosmo.Dataset` transformation that can be described without
in-memory data has a message, discriminated by its ``kind``.

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
