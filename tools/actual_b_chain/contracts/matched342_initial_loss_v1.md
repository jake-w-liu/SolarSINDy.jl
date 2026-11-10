# Initial matched342 validation loss and deterministic selection, version 1

This member records the existing initial-reference metric and selection rule. It introduces no new feature, split, comparator, objective or scientific decision gate.

For each independently admitted receipt delay L in {0,1} and horizon h in {4,6}, fit all eight frozen recipes on that cell's full eligible training cohort. The native scalar MSE response is the target Dst minus its exact completed-hour anchor, in nT. No anchor or target is imputed. Encoder parameters use training rows only. Required or optional driver missingness does not remove otherwise eligible training rows.

Validation uses the full separate per-lead cohort, with the existing 168-hour embargo on both target completion and assumed target receipt, inclusive at the boundary. Candidate columns cover the full delay-specific validation union, including h4-only terminal keys. Existing common required-driver persistence routing applies to published predictions; optional missing features and unknown trace age retain their encoded representation. Numerical, ownership, source or codec failures fail the stage and are not converted to persistence.

For finite absolute predictions p_i and targets y_i on the n eligible per-lead rows, define e_i=p_i-y_i, RMSE=sqrt(sum(e_i^2)/n), MAE=sum(abs(e_i))/n and bias=sum(e_i)/n. Units are nT. Counts and the exact row identities accompany every metric.

Choose the first exact RMSE minimum in the fixed grid order: depth (4,6), rounds (200,400), bins (64,255), with bins varying fastest. Update the selected index only for a strict smaller finite RMSE; exact ties retain the earlier index. No epsilon changes selection. Stored-versus-recomputed metric verification retains the existing bound 5e-12*max(1,abs(expected)); that verification tolerance is not a tie tolerance.

This rule selects only the initial matched-reference model for its own delay and lead. It grants no joint primary-method selection, exposed-development or confirmation scoring, final refitting, calibration or serving authority. Governing sources are the existing DST mechanism candidate contract, fixed ordered342 feature contract, and unchanged BReferenceAdmission metric and strict-minimum implementation.
