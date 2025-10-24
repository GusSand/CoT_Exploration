# Token Naming Convention

## User-Facing Numbering: 1-6

For all **visualizations, documentation, and user-facing output**, continuous thought tokens are numbered **1 through 6**.

### Rationale
- More intuitive for readers (first token = Token 1, not Token 0)
- Aligns with natural language conventions
- Clearer for publication and presentations

## Internal Code: 0-5

Internally, code uses **0-indexed arrays** (tokens 0-5) for compatibility with:
- Python list/array indexing
- PyTorch tensor indexing
- Existing CODI codebase conventions

## Conversion

When displaying results:
```python
# Internal position (0-5)
internal_position = 0

# Display position (1-6)
display_position = internal_position + 1

print(f"Token {display_position}: ...")  # Shows "Token 1: ..."
```

## Key Findings (100-problem experiment)

Using the 1-6 numbering convention:

**Critical Tokens Identified**:
- **Token 6** (last): Most critical (70-80% accuracy when kept alone)
- Tokens 2-3: Moderately critical (~60%)
- Tokens 1, 4-5: Less critical (<50%)

**Paper Claim (middle tokens z₃, z₄ special)**:
- z₃ = Token 4 (1-indexed): NOT critical (<50%)
- z₄ = Token 5 (1-indexed): NOT critical (<50%)
- **Claim refuted**: Token 6 (last) is most critical, not middle tokens

---

**Last Updated**: 2025-10-24
**Experiment**: Token Threshold & Criticality (100 problems running)
