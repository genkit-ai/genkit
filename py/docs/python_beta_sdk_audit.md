# API Audit: Resolved Decisions

The documentation audit surfaced various API issues. The items below had clear Pythonic answers and are resolved. The remaining issues — streaming, public API surface, output configuration, async support, method signatures, and class structure — are open design questions covered in [PYTHON_API_REVIEW.md](./PYTHON_API_REVIEW.md).

---

## Keyword-only arguments

All public methods currently accept positional arguments. Nothing prevents `ai.generate("gemini", "Hi", None, None, ["search"])` — five positional args where the middle three are just filling slots. This is the single most common source of fragile call sites.

**Decision:** Every public method gets a `*` marker after `self`. At most one positional argument is allowed (e.g., `input` on prompt `__call__`). Everything else is keyword-only.

```python
# Before — positional abuse possible
ai.generate("gemini", "Hi", None, None, ["search"])

# After — every argument named
ai.generate(model="gemini", prompt="Hi", tools=["search"])
```

This is standard Python convention. OpenAI, Anthropic, and most modern Python APIs enforce keyword-only arguments on methods with more than 2-3 parameters.

## Decorator shorthands

`@ai.tool()`, `@ai.flow()` exist alongside imperative `define_*` methods. App developers use decorators; plugin authors use the imperative API. This also resolved the handler signature discoverability issue — decorators make expected signatures clear through type hints, while the imperative `define_*` methods accept generic callables with no signature guidance.

^^ Need decorators for other primitives as well.

## Part constructor

Runtime testing revealed that `Part(text="hello")` works via Pydantic's union parsing — `Part` is a `RootModel[Union[TextPart, MediaPart, ...]]` and Pydantic resolves the correct variant from keyword arguments. The verbose `Part(root=TextPart(text="hello"))` form also works but adds no value. Samples use the verbose form.

**Decision:** Bless the shorthand as the documented pattern. Both forms produce identical objects.

```python
Part(text="hello")                                              # blessed
Part(media=Media(url="https://...", content_type="image/png"))  # blessed
```

## RetrieverResponse iterability

`RetrieverResponse` has a `.documents` field but doesn't implement Python's sequence protocol. The audit found 9 occurrences of code trying to iterate over the response directly — the most common single error pattern.

**Decision:** Implement `__iter__`, `__len__`, `__getitem__` delegating to `self.documents`.

```python
# Before — must access .documents
for doc in response.documents:

# After — response is directly iterable
for doc in await ai.retrieve(retriever=my_retriever, query=query):
    print(doc.text)

len(response)    # number of documents
response[0]      # first document
```

This follows the Python convention that collection-like objects should implement the sequence protocol. `RetrieverResponse` is conceptually a list of documents with metadata — it should behave like one.

## response.media property

JS has `response.media` for image generation responses. The audit found 5 occurrences of code using this property — all runtime errors in Python. Users currently have to navigate `response.message.content[0].media`.

**Decision:** Add a `response.media` convenience property on `GenerateResponseWrapper`.

```python
response = await ai.generate(model="googleai/imagen3", prompt="a cat")
image = response.media  # Media | None
```

## Type consolidation

Two nearly-identical types existed: `BaseDataPoint` (generic) and `BaseEvalDataPoint` (evaluator-specific). The audit found samples using them interchangeably.

**Decision:** Merge into `BaseEvalDataPoint`. Remove `BaseDataPoint` from the public API.

## Public API cleanup

Several symbols were in the public `__all__` that don't belong:

- **`tool_response`** — only 3 sample usages. JS and Go use a method on the tool instance.
- **`dump_dict` / `dump_json`** — internal serialization utilities.
- **`get_logger`** — thin wrapper around `logging.getLogger("genkit")`. Python developers know the stdlib.
- **`GenkitRegistry`, `FlowWrapper`, `SimpleRetrieverOptions`** — internal implementation types.

^^ Need to audit __all__ in all packages and see what comes up. 

## Evaluator API

The evaluator API (`GenkitMetricType`, `MetricConfig`, `PluginOptions`) has its own design issues — the audit found the API shape diverges significantly from what the naming suggests. Not addressed in this review; flagged for separate follow-up.

^^ Follow up
