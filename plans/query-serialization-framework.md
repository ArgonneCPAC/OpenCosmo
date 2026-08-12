# Query Serialization Framework Plan

## Goals

Build a transport-independent serialization framework for OpenCosmo queries that:

- Uses Pydantic v2 to describe and validate the wire schema.
- Represents composable operations on `Dataset` and concrete collection objects.
- Supports multiple operations applied sequentially.
- Requires callers to state the exact object type they expect to operate on.
- Rejects object type mismatches before executing any operation.
- Carries a caller-managed UUID so an integrating framework can identify the target
  object among multiple managed objects.
- Returns a typed success or error envelope.
- Provides a serializable, shallow description of the updated object.
- Provides one centralized function for parsing and executing serialized queries.
- Provides a separate function that describes an object under a caller-supplied UUID.
- Provides a fluent, OpenCosmo-like query builder that produces query models without
  resolving or executing against an object.

The framework does not define a transport layer, service, worker, or execution
environment.

## Decisions

The initial design uses the following decisions:

- Support composable, object-returning transformations rather than literally every
  callable public attribute.
- Require an exact object discriminator.
- Stop at the first failed operation and report its index and type.
- Include a safe, typed expression AST for filters and derived columns.
- Return a typed result envelope rather than raising execution errors to callers.
- Return shallow object metadata that does not materialize science data or recursively
  traverse collections.
- Expose stable error codes and safe messages, without serialized tracebacks.
- Require an explicit wire-schema version from the first release.
- Treat object UUIDs as opaque, externally managed identifiers. The serialization
  framework validates and carries them but does not create, assign, resolve, or mutate
  them.
- Provide one immutable `QueryBuilder` whose chained methods append validated
  operations and whose `to_model()` method returns a `QueryRequest`. It exposes the
  complete query API but rejects operations unsupported by its declared object type.

## Supported Object Types

The caller must specify one of these exact object kinds:

- `dataset`
- `structure_collection`
- `simulation_collection`
- `lightcone`
- `healpix_map`

OpenCosmo does not have one concrete collection base class. `Collection` in
`python/opencosmo/collection/protocols.py` is a structural protocol covering only a
small common API. The executor must dispatch on concrete runtime types.

## Object Identity

Every serialized query and object description carries an `object_id` UUID. This lets a
server or other integrating framework associate requests and descriptions with one of
several OpenCosmo objects under its management.

Identity ownership remains entirely outside this framework:

- The integrating framework creates UUIDs.
- The integrating framework stores the UUID-to-object mapping.
- The integrating framework resolves a request's `object_id` to an OpenCosmo object.
- The integrating framework decides whether a caller is authorized to access that UUID.
- The serialization framework never writes a UUID onto a `Dataset` or collection.
- The serialization framework never reads an identity from an OpenCosmo object.
- The serialization framework never generates, replaces, persists, or registers UUIDs.
- `execute_query` does not verify that the request UUID belongs to the object passed to
  it. That association has already been established by the integrating framework.

Pydantic may validate that `object_id` is syntactically a UUID, but this is wire-format
validation only. It does not imply lookup, ownership, existence, or authorization.

An integrating framework is expected to follow this flow:

1. Deserialize or inspect the query envelope.
2. Resolve `object_id` using its own registry.
3. Apply its own authorization and lifecycle checks.
4. Pass the resolved object and query to `execute_query`.
5. Store the returned object under the same UUID, a new UUID, or no UUID according to
   its own policy.

The executor does not prescribe whether a transformed object replaces the prior object
or is registered as a new resource.

## Version 1 Scope

Version 1 supports public transformations that return a new OpenCosmo object and can
be safely described by a closed data model.

Candidate common operations, where supported by the concrete object, are:

- `bound`
- `filter`
- `select`
- `drop`
- `sort_by`
- `take`
- `take_range`
- `take_rows`
- `with_new_columns`
- `with_units`

Candidate spatial and subtype-specific operations are:

- `with_redshift_range`
- `cone_search`
- `box_search`
- `pixel_search`
- `with_resolution`
- Other composable `HealpixMap` transformations confirmed while building the operation
  support matrix

### Explicitly Excluded

- Materializers such as `get_data`, `get_metadata`, `rows`, `objects`, and
  `get_pixels`
- Arbitrary callable execution through `evaluate`
- Lifecycle and I/O methods such as `open`, `close`, and `make_schema`
- Constructors and context-manager methods
- Inherited mutable `dict` methods exposed by collection implementations
- Deprecated `.data` accessors
- Serialization of arbitrary existing `EvaluatedColumn` instances

These exclusions keep the framework declarative and avoid defining schemas for data
frames, generators, file handles, callbacks, and other execution-environment details.

## Proposed Package Structure

Add a dedicated package rather than extending the HDF5-specific
`opencosmo.io.serial` module:

```text
python/opencosmo/serialization/
    __init__.py
    builder.py
    models.py
    expressions.py
    operations.py
    metadata.py
    execute.py
```

Responsibilities:

- `builder.py`: immutable fluent query builders and `make_query`
- `models.py`: request, result, metadata, and error envelopes
- `expressions.py`: expression Pydantic models, encoding, and reconstruction
- `operations.py`: operation schemas, support matrix, and typed dispatch
- `metadata.py`: shallow object metadata extraction
- `execute.py`: the centralized execution function
- `__init__.py`: curated public exports

If the initial implementation is small, `models.py` and `operations.py` can be
combined. Expression handling should remain isolated because it is independently
complex and security-sensitive.

## Base Model Conventions

Use Pydantic v2 frozen models with unknown fields forbidden:

```python
class SerializationModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )
```

Use explicit discriminated unions for:

- Request object types
- Operations
- Expression nodes
- Results
- Object metadata

Do not use order-dependent trial validation for these unions. Stable discriminator
fields improve validation errors and generated JSON Schema.

## Request Envelope

The root request model should be equivalent to:

```python
class QueryRequest(SerializationModel):
    schema_version: Literal[1]
    object_id: UUID
    object_type: ObjectType
    operations: list[Operation]
```

Example:

```json
{
  "schema_version": 1,
  "object_id": "62da09df-284b-4cf1-b063-6834b8f84e52",
  "object_type": "dataset",
  "operations": [
    {
      "operation": "filter",
      "masks": [
        {
          "kind": "comparison",
          "operator": "gt",
          "left": {
            "kind": "column",
            "name": "fof_halo_mass"
          },
          "right": {
            "kind": "quantity",
            "value": 100000000000000.0,
            "unit": "solMass"
          }
        }
      ],
      "mode": "global"
    },
    {
      "operation": "take",
      "count": 100,
      "at": "start",
      "mode": "local"
    }
  ]
}
```

## Fluent Query Builder

Provide a convenience API for constructing `QueryRequest` models using method chains
that resemble direct OpenCosmo operations:

```python
builder = make_query("dataset", object_id)
builder = (
    builder.select("fof_halo_*")
    .filter(oc.col("fof_halo_mass") > 1e14)
    .take(1000, at="random", mode="global")
)
model = builder.to_model()
```

This is only a model-construction API. It does not:

- Resolve `object_id`
- Access or retain an OpenCosmo object
- Validate column names against a dataset
- Execute operations
- Materialize data
- Send the resulting model anywhere
- Register, replace, or otherwise manage an object

### Construction API

The public factory should be similar to:

```python
def make_query(object_type: ObjectType, object_id: UUID) -> QueryBuilder:
    ...
```

Provide one public `QueryBuilder` class with methods for the union of all supported
operations across the five object kinds. The builder stores its declared `object_type`
and consults the central operation support matrix on every method call.

Each builder method follows the same sequence:

1. Look up `(self.object_type, operation)` in the support matrix.
2. Reject the call immediately if that operation is unsupported for the declared type.
3. Select the object-specific Pydantic operation model from the matrix.
4. Convert supported native Python/OpenCosmo arguments into model inputs.
5. Pass the arguments to the selected Pydantic model for validation.
6. Append the validated operation model to a new builder.

Conceptually:

```python
class QueryBuilder:
    def take(self, n: int, at: str = "random", mode: str = "local") -> QueryBuilder:
        model_type = operation_model_for(self.object_type, "take")
        operation = model_type(n=n, at=at, mode=mode)
        return self._append(operation)
```

Calling a subtype-specific method on the wrong builder fails before model construction
or mutation:

```python
builder = make_query("dataset", object_id)
builder.with_resolution(64)
# UnsupportedQueryOperation: "with_resolution" is not supported for "dataset"
```

The unified builder intentionally trades some static autocomplete precision for a
smaller and more maintainable public API. An IDE may show methods that are not valid for
the builder's declared object type; runtime rejection is immediate and deterministic.

Operation models remain object-specific where method semantics differ. For example,
one public `select` builder method dispatches to `DatasetSelectOperation`,
`StructureCollectionSelectOperation`, or another concrete model based on
`self.object_type`. The builder should use a broad enough Python signature to collect
the supported call forms, while the selected Pydantic model owns detailed argument
validation. Do not duplicate that validation in the builder.

### Builder Semantics

Builders should be immutable value objects:

- Each operation method returns a new builder.
- The prior builder remains unchanged and can be reused as the start of another query.
- Operation order is preserved exactly.
- `object_type` and `object_id` remain unchanged across chaining.
- `to_model()` returns a new or immutable `QueryRequest` containing the accumulated
  operation models.
- Calling `to_model()` has no side effects.

For example:

```python
base = make_query("dataset", object_id).select("fof_halo_*")
small = base.take(100)
large = base.take(1000)
```

`small` and `large` are independent queries, and `base` still contains only the
selection operation.

The builder should store validated operation models rather than deferred Python
callables or raw argument tuples. Each method converts its arguments directly into the
same operation schema accepted by deserialization. This keeps builder-produced and
JSON-produced requests behaviorally equivalent.

### Validation Boundaries

The builder validates everything that does not require an actual OpenCosmo object:

- Operation names and object-type support
- Required arguments and enum values
- Positive or otherwise constrained counts and ranges
- Expression shape and supported operators
- Array shape, dtype, and configured size limits
- Unit syntax
- Region shape

Checks that depend on the target object remain execution-time checks:

- Whether a column exists
- Whether a wildcard matches a column
- Whether units are compatible with target columns
- Whether a spatial index exists
- Whether a requested region overlaps the object
- Whether direct arrays have the target object's required row count

### Builder Inputs

Builder methods should accept the familiar in-process values used by OpenCosmo where
they can be encoded safely:

- Strings and iterables of column names
- `Column`, `DerivedScalarValue`, `ColumnMask`, and `CompoundColumnMask` expressions
  constructed through supported public `oc.col(...)` operations
- Astropy scalar quantities and units
- NumPy arrays and quantity arrays allowed by the operation schema
- Supported OpenCosmo region objects or their Pydantic region models

They may also accept the corresponding serialization Pydantic models, which makes
programmatic composition straightforward and avoids unnecessary decode/re-encode
cycles.

Unsupported values must fail while appending the operation with a clear builder or
validation error. In particular, the builder must reject arbitrary callables,
`EvaluatedColumn`, custom reducers, unknown expression callables, object arrays, and
unrecognized region implementations.

Use one stable error type for support-matrix failures, such as
`UnsupportedQueryOperation`, containing the declared object type and operation name.
Pydantic `ValidationError` should remain available for invalid operation arguments so
builder validation and direct model construction report consistent details.

### Model and JSON Output

The required terminal method is:

```python
def to_model(self) -> QueryRequest:
    ...
```

Callers can use Pydantic's normal APIs on the returned model:

```python
model = builder.to_model()
payload = model.model_dump(mode="json")
json_payload = model.model_dump_json()
```

The builder does not need separate transport-oriented methods such as `send`,
`execute`, or `submit`. A `to_json()` convenience method is unnecessary unless later
usage shows that it adds value beyond Pydantic's API.

## Operation Schemas

Define a dedicated Pydantic model for each operation. Do not use a generic method name
plus arbitrary `args` and `kwargs`; that would weaken schema validation and expose
Python calling conventions as the wire format.

Some methods with the same name have incompatible argument structures. They need
object-specific models, for example:

```text
DatasetSelectOperation
StructureCollectionSelectOperation
SimulationCollectionSelectOperation
LightconeSelectOperation
HealpixMapSelectOperation
```

These can share field models, but their root schemas must preserve each concrete API's
semantics.

### Operation Support Matrix

Create an explicit matrix before implementing handlers. It should state, for each
object kind and operation:

- Whether the operation is supported
- The Pydantic request model
- The executor handler
- The expected output object kind
- Any object-specific restrictions
- Whether execution may eagerly load columns

Unsupported object/operation combinations should be rejected during validation when
possible. Execution must not rely on unrestricted `getattr()` dispatch.

## Expression AST

Implement a closed declarative expression language. It must support both decoding wire
models into OpenCosmo expressions for execution and encoding supported in-process
OpenCosmo expressions into wire models for the fluent builder. Never serialize Python
callables, internal operation functions, reducers, UUIDs, or dependency maps.

The expression model should distinguish:

- Column-producing expressions
- Scalar-producing expressions
- Mask expressions
- Direct array inputs for new columns

### Expression Nodes

Support these node categories:

- Column reference
- Integer or floating-point scalar
- Scalar quantity
- Arithmetic expression
- Unary column function
- Reduction
- Comparison mask
- Membership mask
- Boolean mask
- Array input
- Quantity-array input

Representative type aliases are:

```text
ColumnExpression
ScalarExpression
MaskExpression
NewColumnValue
```

### Arithmetic Operations

Support operations exposed by `Column` and `DerivedScalarValue`:

- `add`
- `subtract`
- `multiply`
- `divide`
- `power`

Reconstruct these using public Python operators on values created by `opencosmo.col`.

### Column Functions

Support:

- `log10`
- `exp10`
- `sqrt`
- `arcsin`
- `arccos`
- `arctan2`

For `log10` and `exp10`, expose a closed enum for supported logarithmic unit
containers. Do not accept class names or import paths.

### Reductions

Support:

- `mean`
- `std`
- `var`
- `min`
- `max`
- `median`
- `sum`
- `quantile`

Validate quantile values within `[0, 1]`.

### Masks

Support:

- `eq`
- `ne`
- `gt`
- `ge`
- `lt`
- `le`
- `isin`
- `and`
- `or`

Reconstruct masks through public comparison operators, `Column.isin`, `&`, and `|`.
Do not instantiate mask internals from serialized callable identifiers.

### Reconstruction

Use a hardcoded operation dispatch table:

```python
column = col(model.name)
result = build(model.left) + build(model.right)
result = build(model.operand).mean()
mask = build(model.left) > build(model.right)
mask = build_mask(model.left) & build_mask(model.right)
```

Do not deserialize:

- Callable names
- Dotted imports
- Lambdas
- Reducers
- UUIDs
- Dependency maps
- Cache state
- Pickles
- `EvaluatedColumn`

The enclosing operation's `mode` controls local or global reducers. Existing
`filter`, `select`, and `with_new_columns` methods already attach reducers.

### Encoding Existing Expressions

The builder example accepts expressions produced by the existing `oc.col(...)` API, so
the expression layer also needs a strict encoder:

```python
def expression_to_model(
    expression: Column | DerivedScalarValue | ColumnMask | CompoundColumnMask,
) -> ExpressionModel:
    ...
```

The encoder should recursively inspect only known OpenCosmo expression node classes
and map exact, allowlisted operation identities to AST enums. It must reject nodes
containing arbitrary callables or states not reachable through the supported public
expression API.

Most arithmetic, comparison, reduction, and unary operations already retain stable
named callable identities or `functools.partial` values that can be matched against an
allowlist. Do not infer operations from `repr`, source text, arbitrary function names,
or Python bytecode.

`CompoundColumnMask` currently stores anonymous lambdas for `and` and `or`, which do
not provide a robust semantic tag after construction. To support builder expressions
such as `(oc.col("x") > 0) & (oc.col("y") < 1)`, make the smallest internal change
needed to retain a stable operation identity, for example by using `operator.and_` and
`operator.or_` instead of anonymous lambdas or by storing an explicit enum. Preserve
the existing evaluation behavior. Do not attempt to distinguish arbitrary lambdas by
inspection.

The required round-trip invariant for supported expressions is:

```text
OpenCosmo expression
    -> expression_to_model
    -> reconstruct expression
    -> equivalent behavior on the same object
```

Exact UUIDs, internal callable objects, producer bindings, and cache state are not part
of expression equivalence.

### Context Validation

Validate expression result categories before execution:

- `filter` requires mask roots.
- Derived `select` values may produce columns or scalars.
- Scalar selections cannot be mixed with ordinary columns or column-producing derived
  values.
- `with_new_columns` cannot accept a scalar-producing root.
- Scalar reductions may be nested inside a column-producing expression.

## Primitive Value Schemas

### Units and Quantities

Represent units as strings and quantities as tagged values:

```json
{
  "kind": "quantity",
  "value": 10.0,
  "unit": "Mpc"
}
```

Parse unit strings with Astropy during reconstruction and convert failures into safe
validation errors.

Represent blanket unit conversions as records rather than JSON objects with unit keys:

```json
{
  "conversions": [
    {
      "from_unit": "Mpc",
      "to_unit": "kpc"
    }
  ]
}
```

### Arrays

Use typed array schemas for `take_rows`, `isin`, and direct `with_new_columns` inputs:

```json
{
  "kind": "array",
  "dtype": "int64",
  "shape": [3],
  "values": [1, 4, 9]
}
```

Quantity arrays add a `unit` field.

Validation should:

- Reject object and structured dtypes initially.
- Validate shape against the value count.
- Restrict rank where the target API requires a one-dimensional array.
- Preserve integer precision.
- Reject non-finite values unless an operation explicitly permits them.
- Enforce configurable element and byte limits.
- Never support NumPy pickle encoding.

### Regions

Add explicit discriminators around the existing spatial model shapes:

- `box`
- `skybox`
- `cone`
- `healpix`

Reconstruct them through the public builders in
`python/opencosmo/spatial/builders.py`.

## Execution API

The primary public entry point should be similar to:

```python
def execute_query(
    obj: Dataset | StructureCollection | SimulationCollection | Lightcone | HealpixMap,
    query: QueryRequest | Mapping[str, Any] | str | bytes,
) -> ExecutionResult:
    ...
```

Execution behavior:

1. Parse and validate the request.
2. Determine the exact concrete runtime type of `obj`.
3. Compare it with `query.object_type`.
4. Return an `object_type_mismatch` error immediately if it differs.
5. Execute operations in order, passing each returned object into the next operation.
6. Stop on the first failure.
7. Return a success result containing the final live object and shallow metadata.
8. Never return a partial object on failure.

Use explicit typed handlers. Do not use unrestricted dynamic method dispatch.

`execute_query` does not resolve or compare `query.object_id`. In particular, it must
not inspect the OpenCosmo object for an ID, mutate it to add an ID, or maintain a
registry. The UUID can be copied into the serializable result for correlation, but that
pass-through does not establish or change identity.

### Object Description API

Provide an additional public function:

```python
def describe_object(
    obj: Dataset | StructureCollection | SimulationCollection | Lightcone | HealpixMap,
    object_id: UUID,
) -> ObjectDescription:
    ...
```

`ObjectDescription` is a Pydantic wire model containing:

- `schema_version: 1`
- The caller-supplied `object_id`
- The exact concrete `object_type`
- The same shallow metadata used by successful query results

Example:

```json
{
  "schema_version": 1,
  "object_id": "62da09df-284b-4cf1-b063-6834b8f84e52",
  "object_type": "dataset",
  "metadata": {
    "type": "dataset",
    "length": 100,
    "dtype": "halo_properties",
    "column_names": ["fof_halo_mass"]
  }
}
```

The function uses the UUID only as an opaque value in the returned description. It
does not assign the UUID to `obj`, register it, check it against object state, or retain
it after returning. Metadata extraction should be implemented once and shared by
`describe_object` and successful query execution.

### Live Object and Wire Result

The success result needs to provide the final in-process OpenCosmo object while keeping
the serialized response portable. Two approaches are possible:

1. A Pydantic result with a live-object field excluded from serialization.
2. A non-wire execution wrapper containing the live object and a Pydantic wire result.

The second approach is cleaner for generated JSON Schema because runtime OpenCosmo
types never appear in the schema. Decide between these during implementation after a
small JSON Schema prototype.

Conceptually, callers receive:

```text
ExecutionSuccess
    object: live updated OpenCosmo object
    response: serializable success model
```

## Result Models

Use a discriminated result envelope:

```text
ExecutionResult = ExecutionSuccess | ExecutionFailure
```

The serialized success response should include:

- `status: "success"`
- `schema_version: 1`
- The request's `object_id`, unchanged, for correlation
- Exact object type
- Shallow object metadata

The serialized failure response should include:

- `status: "error"`
- `schema_version: 1`
- The request's `object_id` when it was successfully parsed; otherwise `null`
- A structured error model

## Error Schema

Example:

```json
{
  "status": "error",
  "schema_version": 1,
  "object_id": "62da09df-284b-4cf1-b063-6834b8f84e52",
  "error": {
    "code": "operation_failed",
    "message": "Column 'missing' was not found.",
    "operation_index": 2,
    "operation": "select",
    "validation_errors": null
  }
}
```

Recommended stable error codes:

- `invalid_request`
- `unsupported_schema_version`
- `object_type_mismatch`
- `unsupported_operation`
- `invalid_operation`
- `expression_error`
- `operation_failed`
- `metadata_error`

Recommended fields:

```python
class QueryError(SerializationModel):
    code: ErrorCode
    message: str
    operation_index: int | None = None
    operation: str | None = None
    validation_errors: list[ValidationErrorDetail] | None = None
```

Pydantic validation details should be normalized to JSON-compatible records containing
fields such as:

- `location`
- `message`
- `type`

Do not serialize tracebacks, local variables, filesystem paths, or raw exception
representations. The original exception can remain available internally for logging or
exception chaining.

## Success Metadata

Return a shallow, subtype-discriminated summary that does not materialize science
data.

The metadata models themselves describe OpenCosmo state and do not own identity.
`ObjectDescription` and query result envelopes add the externally supplied `object_id`
around that metadata. This keeps metadata extraction reusable without implying that
OpenCosmo objects contain UUIDs.

### Dataset Metadata

Include:

- Exact object type
- Row count
- Data type
- Column records containing name, stringified unit, and description
- Metadata-column names
- Unit convention
- Redshift or redshift range
- Compact region type
- Sort column
- Spatial-index availability

### StructureCollection Metadata

Include:

- Exact object type
- Structure count
- Source data type
- Visible keys
- Source property names
- Unit convention
- Redshift or redshift range
- Sort column

Do not recursively call `values()`, `items()`, or child indexing by default. These
accesses can rebuild linked datasets.

### SimulationCollection Metadata

Include:

- Exact object type
- Child count
- Child keys
- Shallow child descriptors containing only safely available type, count, and data
  type

Do not invent an aggregate row count because child lengths can differ.

### Lightcone Metadata

Include:

- Exact object type
- Total row count
- Data type
- Step keys as a list that preserves integer keys
- Column metadata
- Unit convention
- Redshift range
- Sort column

### HealpixMap Metadata

Include:

- Exact object type
- Row or pixel count
- Keys
- Column metadata
- Redshift range
- `nside`
- Low-resolution `nside`
- Ordering
- Full-sky flag

Do not include complete pixel lists.

### Metadata Safety Rules

Metadata generation must not use:

- `repr`
- `get_data`
- `.data`
- `rows`
- `get_metadata`
- `make_schema`
- Full HEALPix region conversion
- Recursive `StructureCollection` child access

Unknown or unavailable values should be `null`, not inferred from private fields.

## Failure Semantics

Execution is fail-fast:

- Operation `N` receives the result of operation `N - 1`.
- If operation `N` fails, later operations are not attempted.
- The error records the operation index and discriminator.
- The last successful object is not returned.
- The original input object's logical view remains unchanged by normal transformations.

This is not a transactional rollback guarantee. OpenCosmo objects may share caches and
handlers, and some query operations can populate shared caches even when their logical
views remain immutable. Document this limitation.

## Security and Resource Limits

The query format is a closed language, not a general Python serialization mechanism.

Required safeguards:

- Never use `eval`, `exec`, pickle, dynamic imports, or dotted callable lookup.
- Allow only explicitly enumerated expression and operation types.
- Cap expression depth and total node count.
- Cap array element count and decoded bytes.
- Cap column-name and unit-string lengths.
- Reject object arrays and arbitrary iterator inputs.
- Validate declared array shape and dtype.
- Validate quantile ranges and other operation-specific numeric bounds.
- Treat global reductions, particularly median and quantile under MPI, as potentially
  expensive operations and document that behavior.
- Do not trust serialized output names, UUIDs, dependency maps, reducers, or cache
  state.

## Testing Plan

Add focused test modules:

```text
test/serialization/
    test_builder.py
    test_models.py
    test_expressions.py
    test_operations.py
    test_metadata.py
    test_execute.py
```

### Schema Tests

- JSON round trips for request, operation, result, and metadata models
- Generated JSON Schema contains discriminators
- Unknown fields are rejected
- `schema_version` is required
- Unsupported versions are rejected
- Unknown object and operation tags are rejected
- Invalid object/operation combinations are rejected
- Integer lightcone keys survive serialization where applicable

### Expression Tests

- Every arithmetic operation
- Every unary function
- Every reduction
- Every comparison and boolean mask
- Column-to-column comparisons
- Quantity literals
- `isin` arrays
- Nested reductions such as `(mass - mass.mean()) / mass.std()`
- Scalar-only `select`
- Rejection of scalar roots in `with_new_columns`
- Expression-depth and node-count limits
- Rejection of callable, import, and pickle-like payloads
- Encoding each supported `oc.col(...)` expression into the expected AST
- Encoding and reconstructing compound `and` and `or` masks without lambda inspection
- Behavioral round trips from OpenCosmo expression to model to reconstructed expression
- Rejection of custom constructor-injected expression callables and `EvaluatedColumn`

Compare reconstructed expressions by applying them to test datasets, not by comparing
internal callable identities or UUIDs.

### Builder Tests

- `make_query` always returns one `QueryBuilder` carrying the requested exact object
  type
- Fluent calls produce operations in the same order as the call chain
- The example `select(...).filter(...).take(...)` produces the expected `QueryRequest`
- Every supported builder method produces the same operation model as direct Pydantic
  construction
- Builders are immutable and can be branched safely
- `object_id` and `object_type` remain unchanged
- `to_model()` is repeatable and side-effect free
- Every operation method is available on the unified builder, but unsupported
  object-type/operation pairs fail immediately through the support matrix
- The support failure identifies both the object type and operation
- Same-named methods dispatch to the correct object-specific Pydantic model
- Pydantic models, rather than duplicated builder logic, validate operation arguments
- Object-independent validation occurs while appending an operation
- Object-dependent validation is deferred until execution
- Native Astropy units, quantities, NumPy arrays, regions, and supported `oc.col(...)`
  expressions encode correctly
- Builder methods reject callables, `EvaluatedColumn`, custom reducers, unknown
  expression operations, and unsafe arrays
- Builder-produced models serialize and validate through the same `QueryRequest` schema
  as externally supplied JSON
- Building a query never calls an OpenCosmo object or any execution function

### Sequential Execution Tests

- Multiple operations execute in order
- Exact object-type mismatch fails before operation execution
- The first failing operation stops the sequence
- Errors contain the operation index and type
- No partial object is returned
- Success contains the final live object and serializable metadata
- Every supported input representation accepted by `execute_query` is tested
- The request UUID is passed through unchanged
- `execute_query` does not resolve, register, compare, or assign UUIDs

### Object Matrix Tests

For each concrete object type:

- Every supported operation succeeds on representative fixtures
- Unsupported operations return a stable error
- Same-named but structurally different operations validate correctly
- Each operation returns the expected concrete object type

### Metadata Tests

- Metadata dumping does not call data materializers
- Units become strings
- Descriptions and missing values are JSON-compatible
- Structure collection summaries do not traverse or rebuild children
- Healpix summaries omit pixel arrays
- Simulation collections do not report misleading aggregate row counts
- `describe_object` returns the supplied UUID and the same metadata schema used by
  successful query execution
- `describe_object` does not mutate the described object or retain identity state

### Security Tests

- Arbitrary module and function paths are rejected
- Object arrays are rejected
- Invalid units produce safe validation errors
- Oversized arrays and expressions are rejected
- Serialized errors do not contain tracebacks or filesystem paths
- NaN and infinity handling is explicit

### Verification Commands

```shell
pytest test/serialization
ruff check python/opencosmo/serialization test/serialization
mypy python/opencosmo/serialization
```

After focused tests, run existing dataset, collection, lightcone, map, filter, derive,
units, and spatial tests.

## Documentation

Add:

```text
docs/source/serialization.rst
docs/source/serialization_ref.rst
```

Document:

- Scope and non-goals
- Exact object discriminators
- Caller-owned UUID identity and integration responsibilities
- Fluent query-builder usage and its non-executing behavior
- Sequential execution semantics
- Complete request and response examples
- Expression AST
- Error codes
- Unsupported callable and materializer methods
- Wire-schema versioning policy
- Cache side effects and lack of transactional rollback
- Security and resource-limit behavior

Add the pages to `docs/source/index.rst` and include a Towncrier feature fragment under
`changes/` when implementing the feature.

## Implementation Order

1. Define the exact operation support matrix for all five object kinds.
2. Prototype the success wrapper and generated JSON Schema to decide how the live
   object is separated from the wire response.
3. Add the shared strict Pydantic base model, externally managed UUID field, enums,
   versioned request/result envelopes, object description, and error models.
4. Implement primitive unit, quantity, array, and region schemas.
5. Implement the closed expression AST, strict OpenCosmo-expression encoder, and
   public-API reconstruction. Replace anonymous compound-mask operations with stable
   semantic identities if needed.
6. Implement one immutable `QueryBuilder` and `make_query`, backed by the central
   support matrix and object-specific operation models rather than deferred calls.
7. Implement `Dataset` operation models and handlers first.
8. Add `Lightcone` and `HealpixMap` handlers, which mostly follow Dataset-style APIs.
9. Add `StructureCollection`-specific nested selection, drop, and unit models.
10. Add `SimulationCollection` handlers with explicit child-targeting semantics rather
   than reproducing ambiguous arbitrary `kwargs` behavior.
11. Implement exact runtime type checking and sequential fail-fast execution.
12. Implement shallow metadata summaries and the standalone `describe_object` API.
13. Verify that UUID handling is pass-through only and does not add identity state to
    OpenCosmo objects or the executor.
14. Add public exports, tests, documentation, and the changelog fragment.
15. Run focused and existing regression suites.

The operation support matrix is the first implementation artifact. It prevents
permissive schemas from accidentally exposing incompatible collection behavior and
ensures that generated API schemas accurately describe executor capabilities.
