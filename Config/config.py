import dynaconf

# load config file into settings
sett = dynaconf.Dynaconf(
    core_loaders=["JSON"],
    merge_enabled=True,
    settings_files=["Config/settings.json"],
)
