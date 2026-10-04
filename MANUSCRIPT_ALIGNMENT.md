# Manuscript Alignment Checklist

The repository implements a restricted released audit trail. It records public
configuration, accountant outputs, and module activation status. It does not
release mask sparsity, record-wise sensitivity allocation, or private
representation norms.

Before submission, replace any manuscript statement claiming that the released
audit records directly contain mask sparsity, noise allocation, or
representation norms. A consistent sentence is:

> The released audit trail records public mechanism settings, privacy-accountant outputs, and policy-module activation status without releasing data-dependent mask, sensitivity, or representation statistics.

The per-class recall paragraph should report the gaps from the non-private MLP
in the correct class order:

> Recall for Infiltration, Botnet, and Web Attack remains within 2.2, 1.7, and 2.1 percentage points of the non-private MLP, respectively.
