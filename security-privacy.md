# Self-Review Questionnaire: Security and Privacy

This living document contains the responses to the [Self-Review Questionnaire: Security and Privacy](https://www.w3.org/TR/security-privacy-questionnaire/) for the [Web Neural Network API](https://www.w3.org/TR/webnn/).

## [Questions to Consider](https://www.w3.org/TR/security-privacy-questionnaire/#questions)

> - 2.1 What information might this feature expose to Web sites or other parties, and for what purposes is that exposure necessary?

This feature exposes the `navigator.ml.getNeuralNetworkContext()` factory that encapsulates the rest of the API surface used to create, compile, and run machine learning networks. The API allows web apps to make use of hardware acceleration for neural network inference.

> - 2.2 Is this specification exposing the minimum amount of information necessary to power the feature?

The API exposes the minimum amount of information necessary to address the [identified use cases](https://www.w3.org/TR/webnn/#usecases) for the best performance and reliability of results.

> - 2.3 How does this specification deal with personal information or personally-identifiable information or information derived thereof?

No personal information is exposed.

> - 2.4 How does this specification deal with sensitive information?

No sensitive information is exposed.

> - 2.5 Does this specification introduce new state for an origin that persists across browsing sessions?

No.

> - 2.6 What information from the underlying platform, e.g. configuration data, is exposed by this specification to an origin?

No information from the underlying platform is exposed directly. An execution time analysis may reveal indirectly the performance of the underlying platform's neural network hardware acceleration capabilities relative to another underlying platform.

> - 2.7 Does this specification allow an origin access to sensors on a user’s device

No.

> - 2.8 What data does this specification expose to an origin? Please also document what data is identical to data exposed by other features, in the same or different contexts.

The API adheres to the same-origin policy.

> - 2.9 Does this specification enable new script execution/loading mechanisms?

No.

> - 2.10 Does this specification allow an origin to access other devices?

This specification enables access to the underlying hardware used to acceleration neural network inference.

> - 2.11 Does this specification allow an origin some measure of control over a user agent’s native UI?

No.

> - 2.12 What temporary identifiers might this this specification create or expose to the web?

No temporary identifiers are exposed.

> - 2.13 How does this specification distinguish between behavior in first-party and third-party contexts?

At the moment, the feature does not distinguish between first-party and third-party contexts. Since the feature gives developers access to hardware accelerated features of the device, we could make it be a [policy controlled feature](https://w3c.github.io/webappsec-permissions-policy/#policy-controlled-feature) similar to WebXR and its [`xr-spatial-tracking` feature identifier](https://immersive-web.github.io/webxr/#permissions-policy).

> - 2.14 How does this specification work in the context of a user agent’s Private Browsing or "incognito" mode?

The feature works the same regardless of whether in-private browsing or incognito mode is active.

> - 2.15 Does this specification have a "Security Considerations" and "Privacy Considerations" section?

Work-in-progress at https://github.com/webmachinelearning/webnn/issues/122

> - 2.16 Does this specification allow downgrading default security characteristics?

No.

> - 2.17 What should this questionnaire have asked?

It asked good questions, in particular, 2.15 was helpful for outlining the concerned section.
