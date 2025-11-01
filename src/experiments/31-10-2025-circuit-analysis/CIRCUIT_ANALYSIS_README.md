# CODI Chain-of-Thought Circuit Analysis

This collection of scripts applies circuit analysis techniques (inspired by the Indirect Object Identification work) to analyze how CODI-LLAMA performs continuous chain-of-thought reasoning.

## Overview

CODI (Continuous Thought) models use a unique architecture where:
1. **Standard attention mechanism**: Tokens attend to previous tokens via KV cache
2. **Direct projection pathway**: Hidden state from token n is projected and fed as input to token n+1

This creates a richer information flow compared to standard transformers, where computation happens through:
- **Attention pathway**: Retrieving information from context
- **Direct projection**: Injecting transformed representations from the previous step

## Analysis Scripts

### 1. `codi_circuit_analysis.py` - Intervention Propagation Analysis (PRIMARY)

**Purpose**: Track how interventions at different CoT positions cascade through the reasoning process.

**Key Features**:
- Intervene at each CoT position (0-6) by modifying hidden states
- Track how interventions propagate to downstream positions
- Measure intervention strength and hidden state perturbations
- Visualize intervention cascades

**Approach**: Similar to "activation patching" in IOI analysis - causally modify representations and observe effects.

**Output**:
- `intervention_propagation_analysis.json` - Detailed intervention data
- `intervention_cascade_visualization.png` - 4-panel visualization showing:
  - Token cascade heatmap
  - Intervention propagation strength
  - Hidden state norm changes
  - Projection layer impact

**Run**:
```bash
python codi_circuit_analysis.py
```

### 2. `codi_attention_analysis.py` - Attention Pattern Visualization (SECONDARY)

**Purpose**: Visualize and analyze attention patterns at each CoT position.

**Key Features**:
- Extract attention weights from all layers and heads
- Show what each CoT position attends to (question vs previous CoT)
- Analyze layer-wise attention patterns
- Examine attention head diversity

**Approach**: Similar to attention pattern analysis in IOI - identify where information flows via attention.

**Output**:
- `attention_patterns.json` - Raw attention data
- `attention_overview.png` - Heatmap of average attention patterns
- `attention_distribution.png` - Bar chart showing attention to question vs CoT
- `attention_layerwise_cot{N}.png` - Layer-wise attention for middle CoT position
- `attention_head_diversity.png` - Per-head attention patterns at key positions

**Run**:
```bash
python codi_attention_analysis.py
```

### 3. `codi_pathway_decomposition.py` - Direct Projection vs Attention (TERTIARY)

**Purpose**: Decompose information flow contributions from direct projection vs attention mechanism.

**Key Features**:
- Ablate projection pathway to measure its necessity
- Compare tokens with vs without projection
- Quantify relative contributions of each pathway
- Measure layer-by-layer processing depth

**Approach**: Similar to "component ablation" in IOI analysis - knock out components to measure their contribution.

**Output**:
- `pathway_comparison.json` - Ablation study results
- `information_flow_decomposition.json` - Decomposed contributions
- `pathway_decomposition.png` - 4-panel visualization showing:
  - Effect of ablating projection (token changes)
  - Projection vs attention contributions
  - Relative importance ratio
  - Layer progression analysis

**Run**:
```bash
python codi_pathway_decomposition.py
```

### 4. `codi_full_circuit_analysis.py` - Comprehensive Circuit Diagram (MASTER)

**Purpose**: Unified analysis combining all techniques to create a comprehensive circuit understanding.

**Key Features**:
- Complete circuit trace capturing all information
- Unified circuit diagram showing all pathways
- Logit lens analysis (top-k predictions at each position)
- Quantitative summary of circuit behavior

**Approach**: Integrates all analysis techniques into a single comprehensive view.

**Output**:
- `codi_circuit_diagram.png` - Comprehensive diagram showing:
  - Token flow through CoT positions
  - Attention arrows (to question and previous CoT)
  - Direct projection pathways (red arrows)
  - Logit lens predictions table
  - Information flow metrics
  - Attention distribution
  - Layer processing depth
- `comprehensive_circuit_trace.json` - Complete trace data
- `circuit_summary.txt` - Textual summary of key findings

**Run**:
```bash
python codi_full_circuit_analysis.py
```

## Circuit Analysis Techniques Applied

### From IOI (Indirect Object Identification) Analysis:

1. **Direct Logit Attribution**: Decompose model outputs into per-component contributions
2. **Logit Lens**: Examine what the model "knows" at intermediate layers/positions
3. **Activation Patching**:
   - **Denoising**: Show which components are sufficient for recovery
   - **Noising**: Show which components are necessary
4. **Attention Pattern Analysis**: Identify where information flows
5. **Component Ablation**: Remove components to measure their importance

### Adapted for CODI:

1. **Intervention Propagation**: Track how modifications cascade through CoT (activation patching adapted for continuous thought)
2. **Pathway Decomposition**: Separate attention vs direct projection contributions (unique to CODI)
3. **Position-wise Analysis**: Analyze each CoT step as a computational node in the circuit
4. **Multi-modal Information Flow**: Track both attention-based retrieval and projection-based transformation

## Key Differences from Standard Transformers

| Aspect | Standard Transformer | CODI |
|--------|---------------------|------|
| Information Flow | Only attention | Attention + Direct Projection |
| Token Dependency | Via attention only | Via attention + explicit state transfer |
| Analysis Focus | Attention patterns | Both attention AND projection pathways |
| Circuit Complexity | Single pathway | Dual pathways (requires decomposition) |

## Interpretation Guide

### Understanding the Circuit Diagrams

**Blue arrows**: Attention to original question
- Thicker = stronger attention
- Shows information retrieval from input

**Purple dashed arrows**: Attention to previous CoT tokens
- Shows how model uses its own reasoning steps
- Can indicate iterative refinement

**Red solid arrows**: Direct projection pathway
- Shows explicit state transformation
- Thickness indicates magnitude of projection change
- This is unique to CODI!

**Box colors**:
- Light blue = Question input
- Light yellow = Number tokens (computational results)
- Light green = Operation tokens (+, -, etc.)

### Key Metrics

1. **Projection Contribution**: How much the projection layer changes representations
   - High values = strong transformation
   - Compare to attention contribution

2. **Attention Distribution**: Where the model "looks"
   - High attention to question = retrieving input info
   - High attention to CoT = using intermediate results

3. **Layer Processing Depth**: How much layers transform representations
   - High values = deep processing
   - Shows computational intensity

4. **Intervention Propagation**: How far effects cascade
   - Strong cascade = tight coupling between positions
   - Weak cascade = independent computation

## Example Usage

```bash
# Run all analyses in sequence
python codi_circuit_analysis.py          # Intervention propagation
python codi_attention_analysis.py        # Attention patterns
python codi_pathway_decomposition.py     # Pathway decomposition
python codi_full_circuit_analysis.py     # Comprehensive diagram

# Results will be in ./circuit_analysis_results/
```

## Dependencies

All scripts require:
- PyTorch
- transformers
- peft
- numpy
- matplotlib
- seaborn
- datasets
- python-dotenv
- huggingface_hub

And access to:
- CODI-LLAMA model at `/workspace/CoT_Exploration/models/CODI-llama3.2-1b`
- CODI source code at `/workspace/CoT_Exploration/codi`

## Research Questions Answered

1. **How do interventions propagate through CoT?**
   → See intervention cascade visualization - shows which positions are causally connected

2. **Where does the model attend during reasoning?**
   → See attention pattern analysis - quantifies attention to question vs previous CoT

3. **Which pathway is more important: projection or attention?**
   → See pathway decomposition - compares contribution magnitudes

4. **What is the complete circuit for CoT reasoning?**
   → See comprehensive circuit diagram - integrates all findings

## Future Extensions

Potential additions:
- [ ] Multi-example analysis (aggregate patterns across many questions)
- [ ] Head-specific attribution (which attention heads matter most?)
- [ ] Neuron-level analysis (individual neuron contributions)
- [ ] Comparison with standard transformers (GPT-2, etc.)
- [ ] Layer ablation studies (which layers are critical?)
- [ ] Position-specific intervention strategies (where to intervene most effectively?)

## Citation

This analysis is inspired by:
- Indirect Object Identification circuit analysis (ARENA 3.0)
- Mechanistic interpretability techniques from Anthropic and others
- Applied to CODI (Continuous Thought) architecture

## Contact

For questions or clarifications about the analysis, please refer to the generated visualizations and JSON output files.
