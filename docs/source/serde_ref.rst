Serialization
=============

The :mod:`opencosmo.serde` package describes operations on OpenCosmo objects as
validated, JSON-serializable messages. A message records one call to a public
method. It can be sent over the wire, validated on the other side, and applied to
a live object.

Transformations on :class:`opencosmo.Dataset`, :class:`opencosmo.Lightcone`, and
:class:`opencosmo.StructureCollection` that can be described without in-memory
data have generated messages, discriminated by their ``kind``. The live target
determines which message union is used: for example, ``take`` parses as a
``DatasetTakeMessage`` for a dataset, a ``LightconeTakeMessage`` for a lightcone,
and a ``StructureCollectionTakeMessage`` for a structure collection. Parsed
messages belong to that target type; applying a message to a target of another
type returns an ``unsupported_message`` error.

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

StructureCollection messages
----------------------------

The generated StructureCollection surface contains thirteen operations:
``filter``, ``select``, ``drop``, ``sort_by``, ``take``, ``take_range``,
``take_rows``, ``bound``, ``with_new_columns``, ``with_units``, ``with_datasets``,
``with_redshift_range``, and ``pixel_search``. Their models are exported from
:mod:`opencosmo.serde` as ``StructureCollection<Operation>Message`` classes,
alongside the discriminated ``StructureCollectionMessage`` union and the
``STRUCTURE_COLLECTION_MESSAGE_TYPES`` tuple.

Fields preserve the public method's parameter names and defaults. Variadic
positional arguments travel as JSON lists, and variadic keyword arguments
travel as objects. In particular, StructureCollection ``select`` uses
``select_args`` and ``select_kwargs``; ``drop`` uses ``drop_args`` and
``columns_to_drop``. These differ from the Dataset payload fields.

A selection may contain a column name, a list of names, a derived expression,
an explicit selection with ``columns`` and/or ``derived_columns``, or nested
dataset selections. For example, this selects halo mass and creates a derived
column in a nested galaxy-particle dataset:

.. code-block:: python

    payload = {
        "kind": "select",
        "mode": "global",
        "select_kwargs": {
            "halo_properties": "mass",
            "galaxies": {
                "galaxy_properties": ["tag", "mass"],
                "galaxy_particles": {
                    "columns": ["mass"],
                    "derived_columns": {
                        "double_mass": {
                            "kind": "arithmetic",
                            "operator": "multiply",
                            "left": {"kind": "column", "name": "mass"},
                            "right": {"kind": "number", "value": 2},
                        }
                    },
                },
            },
        },
    }
    result = execute_message(structures, payload)

Column and dataset names in these examples must exist in the target collection.
For automatic column matching, use ``select_args`` (for example,
``["tag", ["mass"]]``) and expression-valued ``select_kwargs``. An empty nested
object remains a nested selection, whereas ``{"columns": []}`` is an explicit
empty dataset selection. ``columns``, ``derived_columns``, and ``kind`` are
reserved selection keys and must have the corresponding selection or expression
shape. Derived selections and ``with_new_columns`` must produce columns;
scalar literals and reductions cannot be applied as new columns.
An explicit selection must include at least one of ``columns`` or
``derived_columns``. Nested ``derived_columns`` cannot use names reserved by
``Dataset.select`` (currently ``mode``). These shape and name restrictions are
also represented in the exported JSON schema.

Nested drops use the same collection hierarchy, with column names or lists at
the leaves:

.. code-block:: python

    payload = {
        "kind": "drop",
        "columns_to_drop": {
            "galaxies": {"galaxy_particles": ["double_mass"]}
        },
    }

Unit payloads use Astropy unit strings. Both keys and values in blanket
``conversions`` maps are unit strings; per-column settings and nested
collections belong in ``dataset_conversions``. Valid strings are normalized
before serialization and converted to live unit objects on application:

.. code-block:: python

    payload = {
        "kind": "with_units",
        "convention": "physical",
        "dataset_conversions": {
            "halo_properties": {"mass": "kg"},
            "galaxies": {
                "galaxy_properties": {"conversions": {"Msun": "kg"}},
                "galaxy_particles": {"mass": "g"},
            },
        },
    }

The optional top-level ``conversions`` map applies blanket conversions across
the collection and may be ``null``. Within recursive unit settings,
``conversions`` must be a map of unit strings to unit strings (an empty map is
allowed); all other values must be unit strings or recursive settings, never
``null``. ``convention`` is reserved at every nested level, while top-level
``dataset_conversions`` reserves both ``convention`` and ``conversions``.
The exported JSON schema describes these recursive restrictions. Unit compatibility and
whether columns carry units are checked by the live operation.

``with_new_columns`` addresses a nested dataset with a dotted ``dataset`` path,
such as ``"galaxies.galaxy_particles"``, and stores expressions in
``new_columns``. ``filter`` stores masks in ``masks`` and preserves
``on_galaxies`` and ``mode``. ``bound`` stores a region message and optional
``select_by``. ``with_redshift_range`` and ``pixel_search`` require a live
lightcone-backed structure collection. HEALPix query pixels use nested ordering:
``pixels`` must be a nonempty list of non-negative integers below
``12 * nside**2``, and ``nside`` must be a positive power of two.

Messages carry no live arrays, quantities, callbacks, or executable code.
``take_rows`` and ``pixel_search`` send JSON integer lists, converted to
one-dimensional signed ``int64`` arrays at application time. Expressions carry
lazy operations rather than materialized column data, including when the live
``with_new_columns`` API also accepts arrays or quantities.

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
