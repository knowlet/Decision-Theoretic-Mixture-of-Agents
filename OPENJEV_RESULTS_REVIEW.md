# v1.4 result interpretation after completed inference and verified publication

This note uses the published v1.4 data; it does not revise the prespecified questions, costs or methods. The semantic order distinction below is a post-release local recalculation from the archived probe records, not a new model run. `audit_openjev_answer_order.py` reproduces it.

## What the head-to-head does support

The 128-case point losses are 0.108125 for the matched fixed-three empirical selector, 0.127656 for calibrated OpenJev 4B and 0.135469 for raw OpenJev 4B. The calibrated-versus-fixed-three interval crosses zero. Raw-versus-fixed-three's ordinary 95% interval does not cross zero, but its multiple-comparison-adjusted interval does. Neither justifies an unqualified claim that the symbolic selector is semantically superior.

Bellman/static/myopic use 1.5 acquired workers on average and obtain J=0.093125; calibrated fixed-three OpenJev is 0.034531 worse in total declared loss, with the adjusted interval remaining positive. Of that point difference, exactly 0.015 is additional worker acquisitions and 0.019531 is terminal loss. The comparison is a pipeline/resource finding, not a pure model-intelligence test. Selector CPU cost is omitted from J and disclosed separately, not silently assigned a zero real cost.

A prompt-based cumulative-score control has the lowest point estimate, 0.089219. Its superiority against Bellman was not a planned contrast, and it is not a CERA implementation. This is further evidence against presenting full Bellman planning as empirically necessary everywhere.

## A substantial confidence-interface lesson

Calibrating candidate correctness on development data sharply reduces the Brier/ECE diagnostic relative to treating raw option mass as correctness. That raw interpretation was not guaranteed by OpenJev or proprietary Jev. Calibration also changes the defer policy. The 4B raw coverage is 118/128 and calibrated coverage 122/128; observed wrong answers remain 11 in both. This difference explains why calibration can improve the objective without making a stronger claim about new reasoning capability.

## Order effects: action identity versus answer semantics

For 4B, reversing options changes the selected action ID in 31/32 probes but changes the actual normalized answer or deferral in only 7/32. Six change deferral status, and mean terminal loss across these 32 probes is identical before and after reversal. The remaining action changes often select a different worker containing the same answer. Calling this a 97% factual failure rate would be false.

For 0.8B, the counts are 28/32 action changes, 23/32 answer-or-deferral changes and 22/32 deferral changes. Reversal does not uniformly make an outcome worse. These are small, deliberately selected robustness probes and should not be extrapolated to a population failure rate.

## Remaining limitations that materially constrain conclusions

Only one neural family, 128 known-benchmark test groups, common archived workers rather than new generation, no human preferences, assumed costs, and CPU precision adaptation were tested. Seventeen test requests differ between capacities in literal last-bit decimal risk serialization. Exact source hashes were recovered, not waived, but this is not byte-identical model input. No evidence supports a proprietary Jev, CERA-MoA or general SOTA ranking.

The strongest contribution remains a reproducible, bounded comparison of estimator freshness, stopping, calibrated terminal decisions and implementation failure modes. More elaborate orchestration is not automatically better; actual semantic and resource contracts determine which comparison is meaningful.
