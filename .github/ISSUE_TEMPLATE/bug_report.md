---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Sanity checks (read this first, then remove this section)**

- [ ] Make sure you're reporting *a bug*; for general questions, please use Discourse or
  Slack.

- [ ] If you're dealing with a performance issue, make sure you **disable scalar iteration**
  (`CUDA.allowscalar(false)`). Only file an issue if that shows scalar iteration happening
  in CUDA.jl or Base Julia, as opposed to your own code.

- [ ] If you're seeing an error message, **follow the error message instructions**, if any
  (e.g. `inspect code with @device_code_warntype`). If you can't solve the problem using
  that information, make sure to post it as part of the issue.

If your bug is still valid, please go ahead and fill out the template below.


**Describe the bug**

A clear and concise description of what the bug is.


**To reproduce**

The Minimal Working Example (MWE) for this bug:

```julia
# some code here
```

<details><summary>Manifest.toml</summary>
<p>

```
Paste your Manifest.toml here, or accurately describe which version of CUDA.jl and its dependencies (GPUArrays.jl, GPUCompiler.jl, LLVM.jl) you are using.
```

</p>
</details>


**Expected behavior**

A clear and concise description of what you expected to happen.


**Version info**

Details on Julia:

```
# please post the output of:
versioninfo()
```

Details on CUDA:

```
# please post the output of:
CUDA.versioninfo()
```


**Additional context**

Add any other context about the problem here.
