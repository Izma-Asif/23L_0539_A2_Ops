import os


def test_project_structure():
    assert os.path.isdir("app")
    assert os.path.isfile("Dockerfile")
    assert os.path.isfile("docker-compose.yml")


def test_env_example_has_no_real_secrets():
    with open(".env.example") as f:
        content = f.read()
    assert "POSTGRES_PASSWORD" in content