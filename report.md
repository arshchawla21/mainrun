# Mainrun LLM Report
> by Arsh Chawla

Each critical component to the design of the LLM has been seperated into a sub-section. Each is accompanied with a light literature review and ablation benchmarks. Implementation details are provided for custom components.

# ToC
- [Tokenizer](Tokenizer)

[Baseline](mainrun/logs/mainrun_2026-06-16T06-50-50.log): 1.753 

## Tokenizer
Tokenization design is critical for downstream performance. We take an emperical approach to determining optimal the design, analysing $fertility$, $utilisation$ etc. for a range of custom and off-the-shelf encodings.