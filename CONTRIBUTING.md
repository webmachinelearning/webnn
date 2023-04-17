# Contributing to the Web Neural Network API
The Working Group welcomes contributions as an issue feedback or a pull request. For contributions aimed at making changes in the specification via a pull request, the following guidelines and best practices should be followed.

## Goal
Members of the Working Group contribute their work and time on a volunteering basis. Our goal is to provide a process that expedites proposing and reviewing changes to the specification within the scope of the [charter](https://www.w3.org/2023/04/web-machine-learning-charter.html) that results in clear and consistent documentation while reducing the opportunities for accidental mistakes and misinterpretation.

## Type of change
Before proposing a change, consider if it falls into one of these categories:

- *Stylistic change*. Change is content-neutral as it affects the visual appearance of the specification without causing a material impact on the content. For example, adding a new section styled by a new stylesheet descriptor isn't considered a stylistic change.
- *Wording change*. Change addresses specific wording without significantly affecting the overall meaning and direction of the content. The purpose of a wording change (sometimes referred to as "editorial change") is to improve the readability of or to clarify complex concepts in the specification.
- *Bug fix*. Change addresses issues in the specification as raised in at least one of GitHub's Issues with sufficient details.
- *New content*. Change introduces new sections to the specification, such as a proposed new operation or an additional method of an interface. Similar to a bug fix, a new content change should be accompanied by a GitHub Issue or other publicly-disclosed artifact that details a need and the proposed new content as a remedy.

It is strongly recommended that a pull request represents a change of just one of these stated categories. For instance, it is not advisable for a pull request that contains a bug fix or new content to also accompany a stylistic change to an unrelated section; or a stylistic change that also accompanies some wording changes. Reasonable specific exceptions, such as opportunistically fixing some misspelled words in a limited amount, are acceptable.

It is also highly desirable that a change to address a certain issue is wholly represented within a single pull request. In other words, a change represented by a pull request is atomic and not distributed across multiple pull requests submitted at different times. This guideline is here to reduce the chance that a specification remains in a transient state for an unspecified period of time.

Change to the document structure, as in rearranging sections within the document without changing the content itself, is considered a stylistic change.

## Process details
Wording change does not require opening a GitHub Issue, as the change in the pull request is self-sufficient. However, wording changes should improve the overall readability and interpretation of the content in an obvious way. Keep in mind that these are subjective qualities, and not everyone may agree to the change. Ultimately, it is up to the consideration of editors whether to accept it.

Similarly, a stylistic change does not necessarily require opening a GitHub Issue. It does, however, require buy-ins from the Working Group in order to proceed. The best way to propose this type of change is to attend one of the bi-weekly Web Machine-Learning Working Group teleconference calls and get a vote. A practical way to reach out to the Working Group to get invited to the teleconference call is to post a GitHub Issue giving a rough explanation of the proposed change and ask to be invited.

Bug fixes and new content changes should proceed as follows:

## Licensing and Working Group participation guidelines
Contributions to this repository are intended to become part of Recommendation-track documents governed by the [W3C Patent Policy](https://www.w3.org/Consortium/Patent-Policy/) and [Software and Document License](https://www.w3.org/Consortium/Legal/copyright-software).

To make substantive contributions to specifications, you must either participate in the relevant W3C Working Group or make a non-member patent licensing commitment. If you are not the sole contributor to a contribution (a pull request), please identify all contributors in the pull request comment.

To add a contributor (other than yourself, that's automatic), mark them one per line as follows:

```
+@github_username
```

If you added a contributor by mistake, you can remove them in a comment with:

```
-@github_username
```

If you are making a pull request on behalf of someone else but you had no part in designing or implementing the change, you can remove yourself from the change using the above syntax.
