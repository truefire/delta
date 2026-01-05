## Best Practices for Using Delta

This page contains a collection of tips for using delta effectively.

### Prompting

- Delta is designed to make intentional edits to a codebase. While it can work with ambiguous prompts or missing information, it will be considerably more effective with concrete prompts. In particular, with precise prompting, Delta can produce code that is very "unsloppy". If you are specific enough, an experienced engineer can often get results very similar to "what they would have written".

### Context & Settings

- Try to keep your codebase composed of "mid-sized" files. Too small and delta has to do an excessive number of edits. Too large and you have to bloat context any time you want to touch that file.
- In addition to files that need changes, include any files that provide critical context (APIs, leaky implementations, etc)
- Keep `Allow REWRITE` off unless you expect to need it -- giving the LLM the ability to explicitly rewrite whole files encourages it to do so, when we typically want it to produce diffs instead.

### Workflow

- The ideal workflow is to select the minimal set of relevant files and use RUN or ASK for a single task.
- ASK -> "Okay do that" -> RUN is a good two-step workflow that gives the LLM more time to think about an issue and lets you review the plan before committing to it.
- PLAN and DIG are niche utility tools -- they're less reliable than vanilla RUN/ASK, but they solve specific problems.
- While Delta supports multi-turn tasks, it's generally not the best way to use the tool.

### Tools

- Your preferred git tooling is the best way to do code review, but the `Review` button is there if you don't want to do that.
- When using DIG, try to use specific terms in your prompt that the LLM can search. Pointing it to the write part of the filesystem can help a lot, especially in a large project.
