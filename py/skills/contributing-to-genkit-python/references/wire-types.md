# Generated wire types

`py/packages/genkit/src/genkit/_core/_typing.py` is generated from the schema shared by all the SDKs (`genkit-tools/genkit-schema.json`) by `py/scripts/schema_to_typing.py`. Don't edit it by hand; the next regeneration will undo your change.

If a type needs to look different in Python, change the generator's tables:

- `TRANSFORMATIONS` renames a type (`Error` becomes `OperationError` so it doesn't shadow the builtin), drops a field (`apiKey` from config, because config shows up in traces), or adds a suffix so a hand-written subclass can extend it.
- `FIELD_TYPE_OVERRIDES` tightens the type of one field.
- `EXCLUDED` is for types we write by hand.

Give each entry a one-line comment explaining why, in terms a user would care about. Then regenerate the file and run `generate_schema_typing --ci` to check it's in sync.
