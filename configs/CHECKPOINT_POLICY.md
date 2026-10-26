# Fixed-last release protocol

The model, optimizer, learning rate, scheduler, image transform, loss, batch sizes, workers, seed and split rules are derived from existing resolved seed42 Full configs. The canonical GSHConv/HICGate code is unchanged.

Historical configurations are retained as sanitized_source_protocol.json. They have test monitoring every 10000 steps, save_top_k=1 and run_test_after_fit=true. They must not be mistaken for a blind-test or test-independent checkpoint protocol.

For the public fixed-last entry point, ghead.yaml explicitly sets intermediate evaluation intervals to null, save_top_k=0 and run_test_after_fit=false. Only the predefined 70000-step last bundle is used by a separate test command. These are documented evaluation/checkpoint policy differences, not changed training hyperparameters. Original defaults in the source project are untouched. Removing intermediate evaluation may change RNG consumption, so exact bitwise replay of historical training is not claimed.

Existing paper aggregates remain formal reference values. Newly repeated fixed-last results are a separate cohort; no promise is made that a fresh run must equal every historical reported aggregate. No test result is used to choose a run, variant, threshold or score calibration.
