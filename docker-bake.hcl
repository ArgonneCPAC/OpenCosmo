group "default" {
  targets = ["dev"]
}

group "dev" {
  targets = ["dev-mpich", "dev-openmpi"]
}

target "dev-mpich" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["docker.io/astropatty/opencosmo:latest"]
}

target "dev-ompi" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["docker.io/astropatty/opencosmo:latest-openmpi"]
  args = {
    MPI_IMPL = "openmpi"
  }
}

target "versioned" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64", "linux/arm64"]
  tags = ["docker.io/astropatty/opencosmo:${GITHUB_REF_NAME}"]
  filters = {
    ref = "refs/tags/[0-9]*"
  }
}

