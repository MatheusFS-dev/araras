# Smart Scheduled Restart Plan

## Goal

Add an optional memory-aware periodic-restart mode to the `monitor` command. The current fixed-interval restart mode must remain available and unchanged. The user chooses which scheduled-restart mode to use; genuine crashes and launch failures keep their existing restart behavior and restart budget in both modes.

## Current Behavior

- `monitor` collects the periodic restart interval in `src/araras/runtime/_monitor_script.py`.
- `run_auto_restart` in `src/araras/runtime/monitoring.py` forwards that interval to `FlagBasedRestartManager`.
- `_wait_for_completion` in `src/araras/runtime/restart_manager.py` returns `scheduled_restart` immediately when the interval deadline expires. This is the existing fixed-interval mode and must remain intact.
- `ProcessMemoryLeakDetector` in `src/araras/runtime/memory_leak.py` already samples combined RSS for the target PID and descendants, and can evaluate a final live sample at a restart boundary.
- Existing leak detection is advisory only. It must not be silently repurposed as the scheduled-restart decision because it requires a long trend window and has different semantics.

## Proposed Behavior

1. Preserve the current fixed-interval mode: when selected, the configured interval expires and the existing `scheduled_restart` path runs without memory polling or new thresholds.
2. Add a separate memory-aware mode. In this mode, the configured interval is the first eligibility point, not an unconditional restart deadline.
3. At each eligible point, poll process-tree RSS at a configurable cadence for a short decision window.
4. Aggregate the samples with a robust value such as the median, then compare the result with an explicit memory threshold.
5. If the threshold is met, return `scheduled_restart` and use the existing replacement, email, graph, report, and scheduled-count paths.
6. If the threshold is not met, keep the same PID running and schedule the next memory-policy check after a configurable polling interval. Do not increment `scheduled_restart_count`, consume `max_restarts`, or send a scheduled-restart email.
7. Continue checking success flags, crash signals, process liveness, interruption, and external stop requests during every wait. Crash or process-death signals must take precedence over either scheduled-restart mode.
8. Define behavior for an unavailable RSS sample explicitly. The preferred default is to log a warning and skip the scheduled restart for that check, avoiding a restart caused by missing telemetry. If fail-open behavior is required for operational safety, make it an explicit configuration choice rather than an implicit fallback.

## Complete Prompt Flow

The shared launch prompts should execute in this order:

1. `Monitor title [default: file stem]:`
	- Blank keeps the per-file title.
2. Show the existing email JSON folder choices.
	- If custom is selected, ask `Custom JSON folder path:`.
3. `Max restarts [default: 10]:`
	- This remains the crash/launch-failure budget and is unaffected by scheduled-restart mode.
4. `Force periodic restart? (Y/n) [default: Y]:`
	- `n`/`no`: disable all scheduled restarts; continue to the memory-leak and report prompts.
	- Blank/`y`/`yes`: ask the existing interval prompt, `Forced restart interval in minutes:`.
5. Ask for the scheduled-restart mode:
	- `Scheduled restart mode: 1) Fixed interval (current behavior) 2) Memory-aware [default: 1]:`
	- `1`: pass the interval through the existing fixed-interval path. Do not ask threshold or polling questions.
	- `2`: keep the interval as the first eligibility point, then ask the memory-policy settings below.
6. For memory-aware mode, ask the threshold using one documented unit:
	- `Process-tree memory threshold ...:`
	- Validate that it is finite and positive (or within the legal range if percentage is selected).
7. Ask the polling cadence:
	- `Memory check interval ...:`
	- Validate that it is finite and positive.
8. Ask the persistence rule:
	- `Required consecutive samples or decision window ...:`
	- Validate that it is finite and positive, or a positive integer for consecutive samples.
9. Continue with the existing prompts:
	- `Detect possible memory leak? (Y/n) [default: Y]:`
	- Memory-leak warm-up prompt when enabled.
	- `Report logs (Y/n) [default: Y]:`

The exact default threshold, unit, polling interval, and persistence value should be selected during implementation and documented in the prompt text. Defaults must not alter the behavior of fixed-interval mode.

Validate all numeric values as finite and positive, and validate percentages within their legal range. Pass the selected mode and smart-policy settings through `_collect_launch_configuration`, `main`, `run_auto_restart`, and `FlagBasedRestartManager`. Keep the existing `force_restart` argument semantics for fixed mode; add separate arguments for smart mode rather than changing the meaning of existing arguments.

The configuration summary should report `Scheduled Restart Mode: Fixed interval` or `Scheduled Restart Mode: Memory-aware`. Only memory-aware mode should display its threshold, polling cadence, and persistence requirement. The existing fixed interval should remain visible in both modes.

## Implementation Steps

1. **Add a small decision policy abstraction** in `src/araras/runtime/restart_manager.py` or a nearby runtime module. Keep it independent of process launching so it can accept sampled RSS values and return a decision. Validate configuration at construction time.
2. **Reuse the existing process-tree RSS helper** from `src/araras/runtime/memory_leak.py`, or extract a shared public helper if the current private helper cannot be reused cleanly. Avoid duplicating root-plus-descendant RSS traversal.
3. **Extend `FlagBasedRestartManager` state and constructor arguments** for the smart-restart policy. Reset policy state whenever a new target PID starts, and stop/reset it during cleanup or when a PID exits.
4. **Add a separate memory-aware wait path** in `_wait_for_completion` or a small helper. Only invoke it when the user selected memory-aware mode; leave the current fixed-interval branch unchanged. Keep the existing boundary crash and liveness checks before each memory decision. Use `time.monotonic()` and the existing `running` flag so the loop remains interruptible.
5. **Preserve existing scheduled-restart handling** after either mode returns a restart decision. The current graph capture, PID replacement confirmation, email notification, report event, and scheduled count should remain unchanged.
6. **Add status messages and report metadata** for memory-aware skipped checks, triggered checks, and unavailable memory measurements. Avoid noisy output by reporting only state changes or a throttled summary. Fixed mode should retain its current messages.
7. **Update user documentation** in `docs/USAGE.md` and, if the public runtime function signature changes, its docstrings. Document both modes and the full prompt flow; clarify that smart scheduling may defer a periodic restart and that crash restarts are unaffected.
8. **Keep `src/araras/TODO.md` as the tracking note**, updating it only after implementation is complete if the project convention is to close completed items there.

## Tests

Add focused unit tests under `tests/` for the policy and manager wait path:

- Below-threshold memory skips the scheduled restart and allows the same process to continue.
- Above-threshold memory triggers exactly one scheduled restart.
- Fixed-interval mode follows the current deadline behavior without invoking memory polling.
- The prompt flow selects fixed versus memory-aware mode and does not ask irrelevant smart-policy questions in fixed mode.
- A transient spike does not trigger a restart when consecutive samples or a decision window are required.
- Missing or invalid memory measurements follow the documented policy and do not crash the monitor loop.
- Invalid interval, threshold, polling, and persistence settings raise clear `ValueError`s.
- A crash signal or dead PID at the scheduled boundary still returns the crash/process-death result before memory evaluation.
- Skipped checks do not change `scheduled_restart_count`, `restart_count`, or the crash restart budget.
- A replacement PID resets the prior PID's sampling state.
- Existing scheduled-restart email/report behavior still occurs only after an actual replacement PID is running and monitored.

Use mocked `psutil` and monotonic time rather than sleeping in tests. Run the focused tests first, then the existing import smoke test.

## Acceptance Criteria

- A running target is not restarted merely because the fixed interval elapsed when memory remains below threshold.
- A target whose process-tree memory remains above threshold for the configured persistence policy is restarted through the existing scheduled-restart path.
- Crash recovery remains independent of smart scheduled restart settings and retains the existing restart budget semantics.
- Memory polling is bounded, interruptible, and does not leave a background sampler running after cleanup.
- CLI prompts, configuration summaries, docs, email/report counts, and tests describe the same behavior and units.
