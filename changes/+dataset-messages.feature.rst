Added ``opencosmo.serde``, which describes ``Dataset`` transformations as validated,
JSON-serializable messages generated from the ``Dataset`` method signatures.
``opencosmo.serde.execute_message`` parses a decoded JSON message for an OpenCosmo
object and applies it, returning a ``MessageError`` when the message is invalid or
unsupported.
