# FAQ

## Working with Git

If your team would like to use Git to share code with one another, consider reviewing these helpful resources.

* The `git` book: [https://git-scm.com/book/en/v2](https://git-scm.com/book/en/v2)
* Concise `git` reference: [https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet](https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet)
* A simple `git` workflow to help you and your team collaborate: [https://gist.github.com/jbenet/ee6c9ac48068889b0912](https://gist.github.com/jbenet/ee6c9ac48068889b0912)

## How to Submit an Issue on Gitlab

For general questions, you can reach out the \#Help channel on the ATD2025 slack or message the challenge administrators.

However, if you encounter a particular bug in the `atd2025` code base that you believe should be resolved, you may submit an issue to the `atd2025` issue tracker. Please refer to [these instructions](https://docs.gitlab.com/ee/user/project/issues/) to learn how to submit issues in Gitlab.


## How can I reinstall our team's conda environment from scratch?

From your terminal run,
```
mamba deactivate  # deactivate the atd2025 environment
mamba env remove --name atd2025  # entirely remove the environment
mamba env create -f environment.yml  # recreate the environment
```

## The `atd2025` library has updated since I first installed it. How can I get the changes?

With your team's conda environment active, run

```
pip install --upgrade atd2025
```
