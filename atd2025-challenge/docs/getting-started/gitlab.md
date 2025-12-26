# Gitlab Authentication

Because `atd2025` is stored within a private repository, you will need to authenticate in order to install the package from the ATD2025 package index.

## Create a Personal Access Token for installing packages from the `atd2025` Gitlab Package registry

Visit [this URL](https://gitlab.com/-/user_settings/personal_access_tokens) to create a Personal Access Token. Give it a name, and grant it at least `read_api` permissions.

## Configure `pip`

First, create a file called `pip.conf` (if using Linux or Mac OSX) or `pip.ini` (if Windows).

```ini
[global]
extra-index-url = https://__token__:your-personal-access-token@gitlab.com/api/v4/groups/97852170/-/packages/pypi/simple
```

Modify the `extra-index-url` to replace `your-personal-access-token` with the personal access token you generated above.

After, store the `pip.conf` (or `pip.ini`, if Windows) in the [appropriate location on your system](https://pip.pypa.io/en/stable/topics/configuration/#location).

On Linux or MacOS, this file should be located at `~/.config/pip/pip.conf`.

Otherwise, if on Windows, you can place the file at `%APPDATA%\pip\pip.ini`
