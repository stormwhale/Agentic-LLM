DeepEval 3 phases of evaluations:
1) Phase 1: Synthetic question generation using DeepSeek-V3.2 API. Requires manual review of question qualities.
2) Phase 2: Feed and extract responses from SMC or other models for evaluation
3) Phase 3: Evaluate the generated outputs for:
   a) Tool Calling Accuracy
   b) Argument correctness (GEval)
   c) Answer relevance (GEval)
   d) Faithfulness (GEval)

The final report and the overall metrics will be saved in "generated_reports" folder in the program directory

*Note: GEval means the metric is a custom metric that gets evaluated by user preferred LLM as Judge.
