[![build](https://github.com/webmachinelearning/webnn/actions/workflows/auto-publish.yml/badge.svg)](https://github.com/webmachinelearning/webnn/actions)

![](https://webmachinelearning.github.io/logos/webnn/logo-webnn-white.png)

# Web Neural Network API

This repository hosts the [Web Neural Network API](https://www.w3.org/TR/webnn/)
being worked on in the
[Web Machine Learning Working Group](https://www.w3.org/groups/wg/webmachinelearning).

To join this Working Group, please see these [instructions](https://webmachinelearning.github.io/community/#join).

## Development

This section describes the process for building this specification on your local computer for development and testing purposes. You may want to do this before you submit a [Pull Request](#pull-requests) to catch possible build issues early.

The first step is to clone the repo:

```
prompt> gh repo clone webmachinelearning/webnn
```

Next, install the makefile dependencies:

```
prompt> cd tools
prompt> npm install
prompt> cd ..
```

Now you can choose between the two options:

- [Install Bikeshed locally](https://speced.github.io/bikeshed/#install-final) (works offline, faster build) and run:

```
prompt> make
```

- Or use the [Bikeshed HTTP API](https://speced.github.io/bikeshed/#remote) (requires internet, slower build):

```
prompt> make online=1
```

In both the cases, the makefile runs [tools](tools) to check the source formatting and catch common errors, and then uses Bikeshed to convert the `index.bs` source file to an `index.html` output file.

If the build is clean you can open the `index.html` file in your browser to view it in all its glory! If you're happy with what you see, congratulations! You can proceed to submit a [Pull Request](#pull-requests).

## Pull Requests

First, please read the [contributing guidelines](CONTRIBUTING.md).

The spec is written using [Bikeshed](https://speced.github.io/bikeshed/) spec-generating tool. Please refer to the [Bikeshed Documentation](https://speced.github.io/bikeshed/) for the Bikeshed syntax and [coding conventions](docs/SpecCodingConventions.md) specific to this specification.

To propose a change to the specification, you are only required to update the `index.bs` source file authored in Bikeshed syntax.

Before you submit a PR, it is recommended to test your changes locally (see [Development](#development)). If your local build is clean, you can proceed to submit the PR. If you want to submit a WIP PR that still contains errors or is incomplete, please [convert the PR to a draft](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request#converting-a-pull-request-to-a-draft). As a convention, please [link the PR to an existing issue](https://graphite.dev/guides/how-to-link-pull-requests-to-github-issues) for [non-editorial changes](CONTRIBUTING.md#type-of-change).

After you submit the PR, it is first automatically checked with [GitHub Actions CI/CD](.github/workflows/auto-publish.yml) for common issues and then reviewed by group participants. After adequate review and approvals, the PR is merged and a new version of the specification is deployed at https://www.w3.org/TR/webnn/ ([history](https://www.w3.org/standards/history/webnn/)).