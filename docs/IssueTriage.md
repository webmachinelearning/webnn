# Triage Guidance for WebNN Repo

## Label Usage

Labels are used for:

- Understanding the status of a specific issue and next steps to resolve it.
- Understanding the scope of work remaining on broad efforts (e.g. aligning with best practices, fixing normative issues, etc)
- Identifying areas to contribute.

The working group chairs and spec editors should regularly review bugs and ensure that labels are accurate, and ensure that issues are getting appropriate attention; for example, scheduling discussion of new feature requests, discussion to resolve outstanding questions, and drawing attention to issues that are ready for a contributor to author a PR.


## Types

Every issue should have one of these issue types, and only rarely more than one.

- https://github.com/webmachinelearning/webnn/labels/bug - a gap or flaw in the specification that will require a normative fix; for example, an algorithm is missing or computes an output incorrectly
- https://github.com/webmachinelearning/webnn/labels/conventions - where the spec does not conform to specification best practices from Web IDL, Bikeshed, Infra, etc.
- https://github.com/webmachinelearning/webnn/labels/use%20case - a new use case for the API that should be documented or considered; may spawn other issues
- https://github.com/webmachinelearning/webnn/labels/process - a meta issue about how the specification is evolved; for example, a discussion of issue labels
- https://github.com/webmachinelearning/webnn/labels/testing - discussion of test coverage
- https://github.com/webmachinelearning/webnn/labels/feature%20request - suggestion for an addition to the proposed API


## Spec Impact

These broad categories describe the projected impact on the specification and implementation of an issue.

- https://github.com/webmachinelearning/webnn/labels/editorial - spec text or styling could be improved, but does not imply changes that functionally affect the interpretation. See [editorial changes](https://www.w3.org/2023/Process-20231103/#editorial-change).

Other issues are generally assumed to require [substantive changes](https://www.w3.org/2023/Process-20231103/#substantive-change).


## Workstream

WebNN has several workstreams specific to the API proposal. These labels help group related issues and measure progress.

- https://github.com/webmachinelearning/webnn/labels/opset - discussions about the overall operator coverage of WebNN; examples include alignment with other published operator sets, use cases that require multiple new operators, compatibility with implementations, etc.
- https://github.com/webmachinelearning/webnn/labels/operator%20specific - issues regarding the specification of a single operator or small number of operators
- https://github.com/webmachinelearning/webnn/labels/webgpu%20interop - interop between WebNN and WebGPU, e.g. timelines, buffers, devices.
- https://github.com/webmachinelearning/webnn/labels/interop - issues arising from differences between backends
- https://github.com/webmachinelearning/webnn/labels/device%20selection - discussions about device selection mechanisms

## Next Steps

- https://github.com/webmachinelearning/webnn/labels/question - there is outstanding discussion needed on the issue before progress can be made
- https://github.com/webmachinelearning/webnn/labels/good%20first%20issue - issues that do not require significant context for new contributors


## Resolved Issues

These labels can be applied to issues when the issue is closed. This is helpful to capture why the issue was closed if it isn't clear from context.

- https://github.com/webmachinelearning/webnn/labels/duplicate
- https://github.com/webmachinelearning/webnn/labels/invalid
- https://github.com/webmachinelearning/webnn/labels/spam
- https://github.com/webmachinelearning/webnn/labels/wontfix

NOTE: GitHub supports two different actions when closing an issue: "Close as completed (Done, closed, fixed, resolved)" and "Close as not planned (Won't fix, can't repo, duplicate, stale)". The UI is subtle, but contributors are encouraged to select an appropriate resolution to assist with future review of issues, in addition to selecting an appropriate label.


## Timeline

- https://github.com/webmachinelearning/webnn/labels/v2 - issue is not considered a blocker for Proposed Recommendation

Implicitly, all issues not tagged **v2** must be resolved before the specification should advance to the next maturity level.


## Horizontal Reviews

These labels will generally be applied to issues by a W3C horizontal review group or to bring an issue to the attention of this group for feedback. These labels are common across W3C spec repos.

- https://github.com/webmachinelearning/webnn/labels/a11y-needs-resolution - raised by Accessibility Group
- https://github.com/webmachinelearning/webnn/labels/a11y-tracker - bring to attention of Accessibility Group
- https://github.com/webmachinelearning/webnn/labels/i18n-needs-resolution - raised by Internationalization Group
- https://github.com/webmachinelearning/webnn/labels/i18n-tracker - bring to attention of Internationalization Group
- https://github.com/webmachinelearning/webnn/labels/privacy-needs-resolution - raised by Privacy Group
- https://github.com/webmachinelearning/webnn/labels/privacy-tracker - bring to attention of Privacy Group
- https://github.com/webmachinelearning/webnn/labels/security-needs-resolution - raised by Security Group
- https://github.com/webmachinelearning/webnn/labels/security-tracker - bring to attention of Security Group
- https://github.com/webmachinelearning/webnn/labels/tag-needs-resolution - raised by Technical Architecture Group
- https://github.com/webmachinelearning/webnn/labels/tag-tracker - bring to attention of Technical Architecture Group


## Label Administration

If you think a new label should be introduced, an old label retired, or the usage of a label reconsidered, please file a PR modifying this file including the proposed change.
