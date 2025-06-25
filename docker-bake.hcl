group "default" {
  targets = ["dev"]
}

group "dev" {
  targets = ["dev-mpich", "dev-openmpi"]
}

variable "GITHUB_REF_NAME" {
  default = "none"
}

target "dev-mpich" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["docker.io/astropatty/opencosmo:latest"]

}

target "dev-openmpi" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["docker.io/astropatty/opencosmo:latest-openmpi"]
  args = {
    MPI_IMPL = "openmpi"
  }
}

target "versioned" {
  dockerfile = "Dockerfile"
  platforms = ["linux/amd64"]
  tags = ["docker.io/astropatty/opencosmo:${GITHUB_REF_NAME}"]
  filters = {
    ref = "refs/tags/[0-9]*"
  }
}

