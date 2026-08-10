---
name: test-writer
description: How to write pytest tests for modules in this workspace. Load whenever you are about to write or extend tests.
---

# Test writer

When writing pytest tests in this workspace:

- **One test file per module.** `foo.py` lives next to `foo_test.py` (suffix, not prefix).
- **Cover the happy path AND at least one edge case.** Empty input, duplicates, boundary values — pick what matters for the unit under test.
- **Use `pytest.mark.parametrize`** when the same assertion runs over a small table of inputs. Keep IDs descriptive.
- **Name tests `test_<unit>_<scenario>_<expected>`.** Examples: `test_total_empty_cart_returns_zero`, `test_add_duplicate_item_merges_quantities`.
- **Arrange / Act / Assert.** Three clear blocks. No setup hidden in fixtures unless it's reused across at least two tests.
- **Assert behavior, not implementation.** Don't reach into private attributes or count function calls; check the observable result.
- **Imports at module top.** Don't import inside test functions.
