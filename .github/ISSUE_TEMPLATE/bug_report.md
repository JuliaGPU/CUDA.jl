---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Sanity checks (read this first, then remove this section)**
Make sure you're reporting *a bug*; for general questions, please use Discourse.

If you're dealing with a performance issue, make sure you have disabled scalar iteration (`CuArrays.allowscalar(false)`). Only file an issue if that shows scalar iteration happening within Base or CuArrays, as opposed to your own code.

If you're seeing an error message, and that message tells you what to do (eg. `inspect code with @device_code_wanrtype`), make sure to do so first and post the relevant output below.

If your bug is still worth reporting, please go ahead and fill out the template below.

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
The Minimal Working Example (MWE) for this bug:
```julia
# some code here
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Version information (please complete the following information):**
 - Julia:
 - CuArrays.jl:
 - CUDAnative.jl:
 - ...

**Additional context**
Add any other context about the problem here.
