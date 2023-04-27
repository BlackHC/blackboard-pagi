import pynecone as pc

config = pc.Config(
    app_name="trace_viewer",
    db_url="sqlite:///pynecone.db",
    env=pc.Env.DEV,
    frontend_packages=[
        "react-flame-graph",
        "react-object-view",
        "react-json-view-lite",
    ],
)
