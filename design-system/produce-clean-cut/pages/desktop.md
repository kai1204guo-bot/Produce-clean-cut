# Desktop workspace override

This file overrides the project master for the Windows desktop production tool.

## Product and dashboard preflight

- artifact_type: native_ui (cross-platform Tk/Windows desktop)
- implementation_owner: existing Tkinter/ttk application
- target_platform: Windows 10 and Windows 11
- audience: operators processing many short-drama series unattended
- primary_job: discover series folders, validate the queue, start processing, and recover failures
- primary_action: start all pending series
- appearance: light
- exact_text: existing Chinese product copy
- reusable_assets: none
- rights_status: not_applicable
- output_format: editable Python source plus Windows installer
- dashboard_type: operational
- primary_decision: what is running, blocked, complete, or safe to retry
- decision_latency: immediate
- primary_data_shape: queue
- density_class: compact
- time_scope: live
- filter_scope: none
- freshness_policy: live
- drilldown_model: inline
- state_contract: loading, empty, partial, error, permission-denied, complete
- topology: table_first
- data_gaps: native Windows high-contrast runtime behavior is not established by Apple guidance

## Apple-informed design translation

This is an original Windows adaptation, not an Apple-native or Apple-approved UI.

- Register: Professional workspace / operational queue.
- Use Segoe UI and Windows-native controls; do not bundle Apple fonts, symbols, logos, or chrome.
- Keep tables and path details opaque. Do not apply glass or decorative gradients.
- Use spacing, alignment, typography, and separators before cards or shadows.
- One blue tint (`#0078D4`) owns primary actions, selection, and keyboard focus.
- Semantic status colors must always be accompanied by explicit state text.

## Tokens

| Role | Value |
|---|---|
| Window background | `#F5F5F7` |
| Primary surface | `#FFFFFF` |
| Subtle surface | `#F2F2F7` |
| Primary label | `#1D1D1F` |
| Secondary label | `#6E6E73` |
| Separator | `#D2D2D7` |
| Tint / focus | `#0078D4` |
| Tint hover | `#106EBE` |
| Success | `#248A3D` |
| Warning | `#B25000` |
| Error | `#D70015` |
| Spacing rhythm | `4, 8, 12, 16, 24` px |
| Control radius | native ttk geometry |
| Motion | none beyond native control feedback |

## Composition

1. Compact title and live queue summary.
2. Queue toolbar with bulk discovery first, single-series add second, utilities last.
3. Opaque queue table as the primary work surface.
4. Selected-series details grouped below the queue.
5. Persistent status band and one dominant Start action.
6. Episode table fills remaining space.

## State and accessibility contract

- Empty queue explains the next action.
- Scanning uses a stable modal shell and determinate result summary.
- Duplicate series are reported, not silently added.
- Running, partial, failed, and completed states retain text labels and row tags.
- All essential actions remain reachable by keyboard with visible ttk focus.
- Destructive removal requires an explicit selected row and never deletes files.
