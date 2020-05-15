---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Sanity checks (read this first, then remove this section)**
Make sure you're reporting *a bug*; for general questions, please use Discourse.

If you're dealing with a performance issue, make sure you **disable scalar iteration** (`CuArrays.allowscalar(false)`). Only file an issue if that shows scalar iteration happening within Base or CuArrays, as opposed to your own code.

If you're seeing an error message, **follow the error message instructions**, if any (eg. `inspect code with @device_code_warntype`). If you can't solve the problem using that information, make sure to post it as part of the issue.

If your bug is still valid, please go ahead and fill out the template below.

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
The Minimal Working Example (MWE) for this bug:
```julia
# some code here
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Build log**
```
# post the output of Pkg.build()
# make sure the error still reproduces after that.
```

**Environment details (please complete this section)**
Details on Julia:
```
# please post the output of:
versioninfo()
```

Julia packages:
 - CuArrays.jl:
 - CUDAnative.jl:
 - ...

CUDA: toolkit and driver version


**Additional context**
Add any other context about the problem here.
