import pynecone as pc

config = pc.Config(
    app_name="my_little_botty",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
    frontend_packages=[
        "react-object-view",
        "react-json-view-lite",
    ],
)
